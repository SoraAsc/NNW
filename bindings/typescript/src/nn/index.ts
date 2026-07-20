import type { WasmBinding } from "../core/wasm.js";
import { type ActivationLike, resolveActivation } from "../types/index.js";

export type TrainerOptimizer = "sgd" | "adamw";

export interface TrainerConfig {
  optimizer?: TrainerOptimizer;
  batchSize?: number;
  shuffle?: boolean;
  learningRate?: number;
}

/**
 * A feed-forward network living in wasm memory.
 *
 * Build it with a fluent API:
 *
 * ```ts
 * const actor = nnw.createModel(4)
 *   .addDense(64, "tanh")
 *   .addDense(64, "tanh")
 *   .addDense(2, "linear");
 * ```
 *
 * Call `.dispose()` when you're done with it. Agents that reference a model
 * do not take ownership of it.
 */
export class NeuralNetwork {
  private handle: number;
  private disposed = false;

  /** @internal use `createModel` on the NNW instance instead of calling this directly */
  constructor(private readonly wasm: WasmBinding, inputDim: number) {
    this.handle = wasm.raw._nn_create_model(inputDim);
  }

  /** Appends a dense layer. Returns `this` so calls can be chained. */
  addDense(units: number, activation: ActivationLike = "linear"): this {
    this.assertAlive();
    if (!Number.isInteger(units) || units <= 0)
      throw new Error(`Dense layer units must be a positive integer, got ${units}`);
    this.wasm.raw._nn_add_dense(this.handle, units, resolveActivation(activation));
    return this;
  }

  get inputDim(): number {
    this.assertAlive();
    return this.wasm.raw._nn_get_input_dim(this.handle);
  }

  get outputDim(): number {
    this.assertAlive();
    return this.wasm.raw._nn_get_output_dim(this.handle);
  }

  /** Returns a portable copy of all trainable parameters in model order. */
  exportParameters(): Float32Array {
    this.assertAlive();
    const count = this.wasm.raw._nn_get_parameter_count(this.handle);
    const ptr = this.wasm.memory.allocOutput(count);
    try {
      if (!this.wasm.raw._nn_export_parameters(this.handle, ptr, count))
        throw new Error("Failed to export neural-network parameters");
      return new Float32Array(this.wasm.memory.readArray(ptr, count));
    } finally {
      this.wasm.memory.free(ptr);
    }
  }

  /** Replaces model parameters. The target architecture must match. */
  importParameters(parameters: Float32Array | number[]): void {
    this.assertAlive();
    const values = Array.from(parameters);
    const expected = this.wasm.raw._nn_get_parameter_count(this.handle);
    if (values.length !== expected)
      throw new Error(`Parameter count mismatch: expected ${expected}, got ${values.length}`);
    this.wasm.memory.withArrays([values], ([ptr]) => {
      if (!this.wasm.raw._nn_import_parameters(this.handle, ptr, values.length))
        throw new Error("Failed to import neural-network parameters");
    });
  }

  /** Reinitializes all trainable parameters using each layer's initializer. */
  resetParameters(): void {
    this.assertAlive();
    this.wasm.raw._nn_reset_parameters(this.handle);
  }

  /** Runs batched inference and returns a flat sample-major output array. */
  predict(inputs: number[] | Float32Array): Float32Array {
    this.assertAlive();
    if (this.outputDim === 0) throw new Error("Cannot run prediction before adding an output layer");
    const values = Array.from(inputs);
    if (values.length === 0) return new Float32Array();
    if (values.length % this.inputDim !== 0)
      throw new Error(`Input length must be a multiple of ${this.inputDim}`);
    const sampleCount = values.length / this.inputDim;
    const outputLength = sampleCount * this.outputDim;
    const out = this.wasm.memory.allocOutput(outputLength);
    try {
      this.wasm.memory.withArrays([values], ([input]) =>
        this.wasm.call(() =>
          this.wasm.raw._nn_predict(
            this.handle,
            input,
            sampleCount,
            this.inputDim,
            out,
            this.outputDim,
          ),
        ),
      );
      return new Float32Array(this.wasm.memory.readArray(out, outputLength));
    } finally {
      this.wasm.memory.free(out);
    }
  }

  /** @internal raw wasm handle, used by PPOAgent when wiring actor/critic together */
  get id(): number {
    this.assertAlive();
    return this.handle;
  }

  /** Frees the underlying wasm model. Safe to call more than once. */
  dispose(): void {
    if (this.disposed) return;
    this.wasm.raw._nn_free_model(this.handle);
    this.disposed = true;
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error("NeuralNetwork used after dispose()");
  }
}

/** Supervised MSE trainer for a neural network living in wasm memory. */
export class NeuralNetworkTrainer {
  private handle: number;
  private disposed = false;

  constructor(
    private readonly wasm: WasmBinding,
    private readonly model: NeuralNetwork,
    config: TrainerConfig = {},
  ) {
    if (model.outputDim === 0) throw new Error("Cannot create a trainer for a model without layers");
    const batchSize = config.batchSize ?? 4;
    const learningRate = config.learningRate ?? 1e-3;
    if (!Number.isInteger(batchSize) || batchSize <= 0)
      throw new Error(`batchSize must be a positive integer, got ${batchSize}`);
    if (!Number.isFinite(learningRate) || learningRate <= 0)
      throw new Error(`learningRate must be positive, got ${learningRate}`);
    const ptr = wasm.memory.allocUint32(4);
    try {
      // NN_TrainerConfig: size_t epochs, size_t batch_size, int shuffle, float learning_rate.
      wasm.memory.writeUint32(ptr, [1, batchSize, config.shuffle === false ? 0 : 1]);
      wasm.memory.writeFloat32(ptr, 12, learningRate);
      this.handle = wasm.call(() =>
        wasm.raw._nn_create_trainer(model.id, config.optimizer === "adamw" ? 1 : 0, 0, ptr),
      );
    } finally {
      wasm.memory.free(ptr);
    }
  }

  /** Trains one epoch and returns the average MSE loss. */
  train(inputs: number[] | Float32Array, targets: number[] | Float32Array): number {
    this.assertAlive();
    const x = Array.from(inputs);
    const y = Array.from(targets);
    if (x.length === 0) throw new Error("Training inputs cannot be empty");
    if (x.length % this.model.inputDim !== 0)
      throw new Error(`Input length must be a multiple of ${this.model.inputDim}`);
    const sampleCount = x.length / this.model.inputDim;
    if (y.length !== sampleCount * this.model.outputDim)
      throw new Error(`Expected ${sampleCount * this.model.outputDim} target values, got ${y.length}`);
    return this.wasm.memory.withArrays([x, y], ([xPtr, yPtr]) =>
      this.wasm.call(() =>
        this.wasm.raw._nn_train_fit(
          this.handle,
          xPtr,
          sampleCount,
          this.model.inputDim,
          yPtr,
          this.model.outputDim,
        ),
      ),
    );
  }

  dispose(): void {
    if (this.disposed) return;
    this.wasm.raw._nn_free_trainer(this.handle);
    this.disposed = true;
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error("NeuralNetworkTrainer used after dispose()");
  }
}
