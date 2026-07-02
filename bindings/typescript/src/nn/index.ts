import type { WasmBinding } from "../core/wasm.js";
import { type ActivationLike, resolveActivation } from "../types/index.js";

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
 * Call `.dispose()` when you're done with it (PPOAgent disposes its
 * actor/critic models automatically when you dispose the agent).
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