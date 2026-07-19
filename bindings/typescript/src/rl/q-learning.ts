import type { WasmBinding } from "../core/wasm.js";
import type { QLearningAgentConfig } from "../types/index.js";

const MAGIC = 0x3154514e; // bytes "NQT1"
const VERSION = 2;
const V1_HEADER_BYTES = 16;
const HEADER_BYTES = 20;

/** Tabular Q-learning agent backed by the native/WASM NNW implementation. */
export class QLearningAgent {
  private handle: number;
  private disposed = false;
  readonly states: number;
  readonly actions: number;
  private readonly decayPerStep: boolean;
  private readonly epsilonStart: number;
  private readonly epsilonMin: number;
  private readonly epsilonDecay: number;
  private trainingEnabled = true;

  constructor(private readonly wasm: WasmBinding, config: QLearningAgentConfig) {
    if (!Number.isInteger(config.states) || config.states <= 0) throw new Error("states must be a positive integer");
    if (!Number.isInteger(config.actions) || config.actions <= 0) throw new Error("actions must be a positive integer");
    this.states = config.states;
    this.actions = config.actions;
    this.decayPerStep = (config.epsilonDecayInterval ?? "step") === "step";
    this.handle = wasm.raw._rl_create_agent(
      config.states,
      config.actions,
      config.learningRate ?? 0.15,
      config.discountFactor ?? 0.98,
    );
    if (!this.handle) throw new Error("Failed to create Q-learning agent");

    this.epsilonStart = config.epsilonStart ?? 1;
    this.epsilonMin = config.epsilonMin ?? 0.02;
    this.epsilonDecay = config.epsilonDecay ?? 0.00002;
    this.configureExploration(this.epsilonStart);
  }

  chooseAction(state: number): number {
    this.assertState(state);
    this.assertAlive();
    return this.wasm.raw._rl_choose_agent_action(this.handle, state);
  }

  update(state: number, action: number, reward: number, nextState: number, done = false): void {
    this.assertAlive();
    this.assertState(state);
    this.assertState(nextState);
    if (!Number.isInteger(action) || action < 0 || action >= this.actions) throw new Error(`Invalid action: ${action}`);
    this.wasm.raw._rl_update_agent(this.handle, state, action, reward, nextState, done ? 1 : 0);
    if (this.trainingEnabled) {
      if (this.decayPerStep) this.wasm.raw._rl_update_agent_epsilon_step(this.handle);
      else if (done) this.wasm.raw._rl_update_agent_epsilon_episode(this.handle);
    }
  }

  get epsilon(): number {
    this.assertAlive();
    return this.wasm.raw._rl_get_agent_epsilon(this.handle);
  }

  get training(): boolean {
    this.assertAlive();
    return this.trainingEnabled;
  }

  set training(value: boolean) {
    this.assertAlive();
    this.trainingEnabled = value;
    this.wasm.raw._rl_set_agent_training(this.handle, value ? 1 : 0);
  }

  exportQTable(): Float32Array {
    this.assertAlive();
    const count = this.states * this.actions;
    const ptr = this.wasm.memory.allocOutput(count);
    try {
      if (!this.wasm.raw._rl_export_agent_qtable(this.handle, ptr, count)) throw new Error("Failed to export Q-table");
      return new Float32Array(this.wasm.memory.readArray(ptr, count));
    } finally {
      this.wasm.memory.free(ptr);
    }
  }

  importQTable(values: Float32Array | number[]): void {
    this.assertAlive();
    const data = Array.from(values);
    const expected = this.states * this.actions;
    if (data.length !== expected) throw new Error(`Q-table size mismatch: expected ${expected}, got ${data.length}`);
    this.wasm.memory.withArrays([data], ([ptr]) => {
      if (!this.wasm.raw._rl_import_agent_qtable(this.handle, ptr, data.length)) throw new Error("Failed to import Q-table");
    });
  }

  saveCheckpoint(): ArrayBuffer {
    const table = this.exportQTable();
    const buffer = new ArrayBuffer(HEADER_BYTES + table.byteLength);
    const view = new DataView(buffer);
    view.setUint32(0, MAGIC, true);
    view.setUint32(4, VERSION, true);
    view.setUint32(8, this.states, true);
    view.setUint32(12, this.actions, true);
    view.setFloat32(16, this.epsilon, true);
    new Float32Array(buffer, HEADER_BYTES).set(table);
    return buffer;
  }

  loadCheckpoint(buffer: ArrayBuffer): void {
    if (buffer.byteLength < V1_HEADER_BYTES) throw new Error("Q-table checkpoint is truncated");
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== MAGIC) throw new Error("File is not an NNW Q-table checkpoint");
    const version = view.getUint32(4, true);
    if (version !== 1 && version !== VERSION) throw new Error(`Unsupported Q-table checkpoint version: ${version}`);
    const states = view.getUint32(8, true), actions = view.getUint32(12, true);
    if (states !== this.states || actions !== this.actions)
      throw new Error(`Q-table dimensions mismatch: checkpoint ${states}x${actions}, agent ${this.states}x${this.actions}`);
    const headerBytes = version === 1 ? V1_HEADER_BYTES : HEADER_BYTES;
    const expectedBytes = headerBytes + states * actions * Float32Array.BYTES_PER_ELEMENT;
    if (buffer.byteLength !== expectedBytes) throw new Error("Q-table checkpoint has an invalid size");
    this.importQTable(new Float32Array(buffer, headerBytes));
    // V1 stored only the table. Resume old checkpoints with conservative
    // exploration instead of epsilon=1, which makes a loaded policy look lost.
    const restoredEpsilon = version === 1 ? this.epsilonMin : view.getFloat32(16, true);
    this.configureExploration(restoredEpsilon);
  }

  reset(): void {
    this.assertAlive();
    this.wasm.raw._rl_clear_agent_qtable(this.handle);
    this.configureExploration(this.epsilonStart);
  }

  dispose(): void {
    if (this.disposed) return;
    this.wasm.raw._rl_free_agent(this.handle);
    this.disposed = true;
  }

  private assertState(state: number): void {
    if (!Number.isInteger(state) || state < 0 || state >= this.states) throw new Error(`Invalid state: ${state}`);
  }

  private configureExploration(start: number): void {
    if (!Number.isFinite(start) || start < 0 || start > 1) throw new Error(`Invalid epsilon: ${start}`);
    this.wasm.raw._rl_set_agent_policy(this.handle, 1, start);
    if (!this.wasm.raw._rl_set_agent_epsilon_decay(
      this.handle,
      start,
      Math.min(this.epsilonMin, start),
      this.epsilonDecay,
      1,
      this.decayPerStep ? 1 : 0,
    )) throw new Error("Invalid epsilon decay configuration");
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error("QLearningAgent used after dispose()");
  }
}
