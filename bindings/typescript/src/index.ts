import { WasmBinding } from "./core/wasm.js";
import { NeuralNetwork, NeuralNetworkTrainer, type TrainerConfig } from "./nn/index.js";
import { PPOAgent } from "./rl/index.js";
import { QLearningAgent } from "./rl/q-learning.js";
import type { PPOAgentConfig } from "./types/index.js";
import {
  loadCheckpoint,
  resetModels,
  saveCheckpoint,
  type CheckpointModels,
} from "./checkpoint/index.js";

export * from "./types/index.js";
export { NeuralNetwork, NeuralNetworkTrainer, type TrainerConfig } from "./nn/index.js";
export { PPOAgent } from "./rl/index.js";
export { QLearningAgent } from "./rl/q-learning.js";
export { loadCheckpoint, resetModels, saveCheckpoint, type CheckpointModels } from "./checkpoint/index.js";

/**
 * The main handle for the library. Create models and PPO agents from here.
 *
 * ```ts
 * const nnw = await createNNW();
 * const actor = nnw.createModel(4).addDense(64, "tanh").addDense(2, "linear");
 * const critic = nnw.createModel(4).addDense(64, "tanh").addDense(1, "linear");
 * const agent = nnw.createPPOAgent(actor, critic, { actionSpace: "discrete", numEnvs: 8, rolloutSteps: 128, minibatchSize: 64 });
 * ```
 */
export class NNW {
  private constructor(private readonly wasm: WasmBinding) {}

  /** @internal use `createNNW()` instead */
  static async init(): Promise<NNW> {
    const wasm = await WasmBinding.create();
    return new NNW(wasm);
  }

  /** Creates a new empty model with the given input dimension. Add layers with `.addDense(...)`. */
  createModel(inputDim: number): NeuralNetwork {
    return new NeuralNetwork(this.wasm, inputDim);
  }

  /** Creates an MSE trainer for supervised learning. */
  createTrainer(model: NeuralNetwork, config: TrainerConfig = {}): NeuralNetworkTrainer {
    return new NeuralNetworkTrainer(this.wasm, model, config);
  }

  /** Creates a PPO agent wired to the given actor/critic models. */
  createPPOAgent(actor: NeuralNetwork, critic: NeuralNetwork, config: PPOAgentConfig): PPOAgent {
    return new PPOAgent(this.wasm, actor, critic, config);
  }

  /** Creates a tabular Q-learning agent. */
  createQLearningAgent(config: import("./types/index.js").QLearningAgentConfig): QLearningAgent {
    return new QLearningAgent(this.wasm, config);
  }

  /** Serializes a named collection of models into a versioned binary checkpoint. */
  saveCheckpoint(models: CheckpointModels): ArrayBuffer {
    return saveCheckpoint(models);
  }

  /** Loads a checkpoint into already constructed compatible models. */
  loadCheckpoint(buffer: ArrayBuffer, models: CheckpointModels): void {
    loadCheckpoint(buffer, models);
  }

  /** Reinitializes a collection of models from scratch. */
  resetModels(models: CheckpointModels): void {
    resetModels(models);
  }
}

/** Loads the wasm module and returns a ready-to-use `NNW` instance. */
export async function createNNW(): Promise<NNW> {
  return NNW.init();
}
