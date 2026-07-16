import type { NeuralNetwork } from "../nn/index.js";

export type CheckpointModels = Record<string, NeuralNetwork>;

type ModelMetadata = {
  name: string;
  inputDim: number;
  outputDim: number;
  parameterCount: number;
  parameterOffset: number;
};

type CheckpointMetadata = {
  format: "nnw-checkpoint";
  version: 1;
  models: ModelMetadata[];
};

const MAGIC = 0x43574e4e; // bytes "NNWC"
const VERSION = 1;
const FIXED_HEADER_BYTES = 12;

function align4(value: number): number {
  return (value + 3) & ~3;
}

/** Serializes any named collection of constructed NNW models. */
export function saveCheckpoint(models: CheckpointModels): ArrayBuffer {
  const entries = Object.entries(models);
  if (entries.length === 0) throw new Error("saveCheckpoint requires at least one model");

  const seen = new Set<string>();
  const parameterBlocks: Float32Array[] = [];
  let parameterOffset = 0;
  const metadata: CheckpointMetadata = {
    format: "nnw-checkpoint",
    version: VERSION,
    models: entries.map(([name, model]) => {
      if (!name || seen.has(name)) throw new Error(`Invalid or duplicate checkpoint model name: "${name}"`);
      seen.add(name);
      const parameters = model.exportParameters();
      parameterBlocks.push(parameters);
      const item = {
        name,
        inputDim: model.inputDim,
        outputDim: model.outputDim,
        parameterCount: parameters.length,
        parameterOffset,
      };
      parameterOffset += parameters.length;
      return item;
    }),
  };

  const encodedMetadata = new TextEncoder().encode(JSON.stringify(metadata));
  const payloadStart = align4(FIXED_HEADER_BYTES + encodedMetadata.length);
  const buffer = new ArrayBuffer(payloadStart + parameterOffset * Float32Array.BYTES_PER_ELEMENT);
  const view = new DataView(buffer);
  view.setUint32(0, MAGIC, true);
  view.setUint32(4, VERSION, true);
  view.setUint32(8, encodedMetadata.length, true);
  new Uint8Array(buffer, FIXED_HEADER_BYTES, encodedMetadata.length).set(encodedMetadata);

  const payload = new Float32Array(buffer, payloadStart, parameterOffset);
  for (let i = 0; i < parameterBlocks.length; ++i)
    payload.set(parameterBlocks[i]!, metadata.models[i]!.parameterOffset);
  return buffer;
}

/** Loads a checkpoint into already constructed, architecture-compatible models. */
export function loadCheckpoint(buffer: ArrayBuffer, models: CheckpointModels): void {
  if (buffer.byteLength < FIXED_HEADER_BYTES) throw new Error("NNW checkpoint is truncated");
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) throw new Error("File is not an NNW checkpoint");
  if (view.getUint32(4, true) !== VERSION) throw new Error(`Unsupported NNW checkpoint version: ${view.getUint32(4, true)}`);

  const metadataLength = view.getUint32(8, true);
  const payloadStart = align4(FIXED_HEADER_BYTES + metadataLength);
  if (payloadStart > buffer.byteLength) throw new Error("NNW checkpoint metadata is truncated");

  let metadata: CheckpointMetadata;
  try {
    const json = new TextDecoder().decode(new Uint8Array(buffer, FIXED_HEADER_BYTES, metadataLength));
    metadata = JSON.parse(json) as CheckpointMetadata;
  } catch {
    throw new Error("NNW checkpoint metadata is invalid");
  }
  if (metadata.format !== "nnw-checkpoint" || metadata.version !== VERSION || !Array.isArray(metadata.models))
    throw new Error("NNW checkpoint metadata is incompatible");

  const payloadFloats = (buffer.byteLength - payloadStart) / Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isInteger(payloadFloats)) throw new Error("NNW checkpoint payload is misaligned");
  const payload = new Float32Array(buffer, payloadStart, payloadFloats);
  const seen = new Set<string>();
  const imports: Array<{ model: NeuralNetwork; parameters: Float32Array }> = [];

  for (const item of metadata.models) {
    if (!item || typeof item.name !== "string" || seen.has(item.name))
      throw new Error("NNW checkpoint contains an invalid model name");
    seen.add(item.name);
    const model = models[item.name];
    if (!model) throw new Error(`Missing target model for checkpoint entry "${item.name}"`);
    if (model.inputDim !== item.inputDim || model.outputDim !== item.outputDim)
      throw new Error(`Architecture mismatch for model "${item.name}"`);
    if (!Number.isInteger(item.parameterOffset) || !Number.isInteger(item.parameterCount)
        || item.parameterOffset < 0 || item.parameterCount < 0
        || item.parameterOffset + item.parameterCount > payload.length)
      throw new Error(`Invalid parameter range for model "${item.name}"`);
    imports.push({
      model,
      parameters: payload.slice(item.parameterOffset, item.parameterOffset + item.parameterCount),
    });
  }

  // Mutate models only after every entry has passed validation.
  for (const item of imports) item.model.importParameters(item.parameters);
}

/** Reinitializes all supplied models with their layer initializers. */
export function resetModels(models: CheckpointModels): void {
  const entries = Object.entries(models);
  if (entries.length === 0) throw new Error("resetModels requires at least one model");
  for (const [, model] of entries) model.resetParameters();
}
