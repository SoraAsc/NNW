import { WasmBinding } from "./core/wasm.js";
import { NeuralNetwork } from "./nn/index.js";
import { PPOAgent } from "./rl/index.js";
import type { PPOAgentConfig } from "./types/index.js";

export * from "./types/index.js";
export { NeuralNetwork } from "./nn/index.js";
export { PPOAgent } from "./rl/index.js";

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

  /** Creates a PPO agent wired to the given actor/critic models. */
  createPPOAgent(actor: NeuralNetwork, critic: NeuralNetwork, config: PPOAgentConfig): PPOAgent {
    return new PPOAgent(this.wasm, actor, critic, config);
  }
}

/** Loads the wasm module and returns a ready-to-use `NNW` instance. */
export async function createNNW(): Promise<NNW> {
  return NNW.init();
}