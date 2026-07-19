/**
 * Low-level wasm plumbing: module loading + raw memory helpers.
 *
 * Nothing in this file is meant to be imported by end users directly —
 * it's consumed by `src/nn` and `src/rl` to build the friendly public API
 * exposed from `src/index.ts`.
 */

export interface EmscriptenModuleLike {
  _malloc: (size: number) => number;
  _free: (ptr: number) => void;

  _nn_create_model: (inputDim: number) => number;
  _nn_free_model: (model: number) => void;
  _nn_add_dense: (model: number, units: number, activation: number) => void;
  _nn_get_input_dim: (model: number) => number;
  _nn_get_output_dim: (model: number) => number;
  _nn_get_parameter_count: (model: number) => number;
  _nn_export_parameters: (model: number, out: number, count: number) => number;
  _nn_import_parameters: (model: number, data: number, count: number) => number;
  _nn_reset_parameters: (model: number) => void;

  _rl_ppo_create_agent: (...args: number[]) => number;
  _rl_ppo_free_agent: (agent: number) => void;
  _rl_ppo_collect_step: (
    agent: number,
    states: number,
    batchSize: number,
    outActions: number,
    outLogProbs: number,
    outValues: number,
  ) => void;
  _rl_ppo_store_transition: (
    agent: number,
    states: number,
    batchSize: number,
    actions: number,
    logProbs: number,
    rewards: number,
    terminals: number,
    values: number,
  ) => void;
  _rl_ppo_train: (agent: number, nextValue: number, batchSize: number, nextTerminal: number) => void;

  _rl_create_agent: (states: number, actions: number, learningRate: number, discountFactor: number) => number;
  _rl_free_agent: (agent: number) => void;
  _rl_choose_agent_action: (agent: number, state: number) => number;
  _rl_update_agent: (agent: number, state: number, action: number, reward: number, nextState: number, done: number) => void;
  _rl_set_agent_policy: (agent: number, policy: number, epsilon: number) => void;
  _rl_set_agent_epsilon_decay: (agent: number, start: number, min: number, rate: number, type: number, perStep: number) => number;
  _rl_update_agent_epsilon_step: (agent: number) => void;
  _rl_update_agent_epsilon_episode: (agent: number) => void;
  _rl_get_agent_epsilon: (agent: number) => number;
  _rl_set_agent_training: (agent: number, training: number) => void;
  _rl_get_agent_training: (agent: number) => number;
  _rl_get_agent_states: (agent: number) => number;
  _rl_get_agent_actions: (agent: number) => number;
  _rl_export_agent_qtable: (agent: number, out: number, count: number) => number;
  _rl_import_agent_qtable: (agent: number, data: number, count: number) => number;
  _rl_clear_agent_qtable: (agent: number) => void;

  /** Present only if nn.mjs was built with -sEXPORTED_RUNTIME_METHODS=getExceptionMessage */
  getExceptionMessage?: (excPtr: number) => string;

  HEAPF32: Float32Array;
  HEAPU32: Uint32Array;
  HEAPU8: Uint8Array;
}

type EmscriptenFactory = (options?: Record<string, unknown>) => Promise<EmscriptenModuleLike>;

/** Loads the compiled wasm module (nn.mjs) relative to this package. */
export async function loadWasmModule(): Promise<EmscriptenModuleLike> {
  const moduleUrl = new URL("../libs/nn.mjs", import.meta.url);
  const factory = ((await import(/* @vite-ignore */ moduleUrl.href)) as { default: EmscriptenFactory })
    .default;
  return factory({
    locateFile: (name: string) => {
      if (name.endsWith(".wasm")) return new URL("../libs/" + name, import.meta.url).href;
      return name;
    },
  });
}

/**
 * Thin RAII-ish helper around the module's linear memory. Every array
 * handed to the wasm side goes through here so allocation/free is never
 * duplicated (and never forgotten) in the higher-level classes.
 */
export class WasmMemory {
  constructor(private readonly mod: EmscriptenModuleLike) {}

  /** Allocates a Float32 buffer in wasm memory and copies `values` into it. */
  allocArray(values: number[]): number {
    const bytes = values.length * Float32Array.BYTES_PER_ELEMENT;
    const ptr = this.mod._malloc(bytes);
    new Float32Array(this.mod.HEAPF32.buffer, ptr, values.length).set(values);
    return ptr;
  }

  /** Allocates an unsigned 32-bit buffer (WASM size_t) and copies values. */
  allocSizeArray(values: number[]): number {
    const bytes = values.length * Uint32Array.BYTES_PER_ELEMENT;
    const ptr = this.mod._malloc(bytes);
    new Uint32Array(this.mod.HEAPU32.buffer, ptr, values.length).set(values);
    return ptr;
  }

  /** Allocates an uninitialized Float32 output buffer of `length` elements. */
  allocOutput(length: number): number {
    return this.mod._malloc(length * Float32Array.BYTES_PER_ELEMENT);
  }

  readArray(ptr: number, length: number): number[] {
    if (ptr === 0) return [];
    return Array.from(new Float32Array(this.mod.HEAPF32.buffer, ptr, length));
  }

  free(ptr: number): void {
    if (ptr !== 0) this.mod._free(ptr);
  }

  /** Runs `fn` with a set of temporary input buffers, freeing all of them afterwards, even on error. */
  withArrays<T>(inputs: number[][], fn: (ptrs: number[]) => T): T {
    const ptrs = inputs.map((values) => this.allocArray(values));
    try {
      return fn(ptrs);
    } finally {
      for (const ptr of ptrs) this.free(ptr);
    }
  }
}

/** Bundles the raw module + memory helper together; passed internally to NeuralNetwork/PPOAgent. */
export class WasmBinding {
  readonly memory: WasmMemory;

  private constructor(readonly raw: EmscriptenModuleLike) {
    this.memory = new WasmMemory(raw);
  }

  static async create(): Promise<WasmBinding> {
    const raw = await loadWasmModule();
    return new WasmBinding(raw);
  }

  /**
   * Runs `fn`, and if the wasm side throws a C++ exception, rethrows it as a
   * regular Error with a readable message instead of the opaque
   * `CppException { excPtr }` emscripten normally surfaces.
   */
  call<T>(fn: () => T): T {
    try {
      return fn();
    } catch (err) {
      throw new Error(this.describeException(err));
    }
  }

  private describeException(err: unknown): string {
    const excPtr = (err as { excPtr?: number } | null)?.excPtr;
    if (typeof excPtr === "number") {
      if (typeof this.raw.getExceptionMessage === "function") {
        return `NNW wasm error: ${this.raw.getExceptionMessage(excPtr)}`;
      }
      return (
        `NNW wasm threw a C++ exception (ptr=${excPtr}) but the build doesn't export ` +
        `getExceptionMessage, so the message can't be decoded here. Rebuild nn.mjs with ` +
        `emcc flag -sEXPORTED_RUNTIME_METHODS=getExceptionMessage (and -fexceptions) to get ` +
        `readable error messages.`
      );
    }
    return String(err);
  }
}
