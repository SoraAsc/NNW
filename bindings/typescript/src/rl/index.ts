import type { WasmBinding } from "../core/wasm.js";
import type { NeuralNetwork } from "../nn/index.js";
import {
  ActionSpace,
  type PPOAgentConfig,
  type PPOStepResult,
  resolveActionSpace,
  resolveOptimizer,
} from "../types/index.js";

const DEFAULTS = {
  optimizer: "adamw" as const,
  learningRate: 3e-4,
  gamma: 0.99,
  gaeLambda: 0.95,
  clipRange: 0.2,
  valueLossCoef: 0.5,
  entropyCoef: 0.01,
  maxGradNorm: 0.5,
  epochs: 4,
};

/**
 * A PPO agent bound to an actor/critic pair.
 *
 * ```ts
 * const agent = nnw.createPPOAgent(actor, critic, {
 *   actionSpace: "discrete",
 *   numEnvs: 8,
 *   rolloutSteps: 128,
 *   minibatchSize: 64,
 * });
 *
 * const { actions, logProbs, values } = agent.collectStep(states, numEnvs);
 * agent.storeTransition(states, numEnvs, actions, logProbs, rewards, terminals, values);
 * agent.train(bootstrapValues, bootstrapTerminals);
 * agent.dispose();
 * ```
 */
export class PPOAgent {
  private handle: number;
  private disposed = false;
  private readonly stateDim: number;
  private readonly numEnvs: number;

  /**
   * Number of floats per env in the `actions` buffer.
   * - discrete: 1 (the sampled action index)
   * - continuous / multiBinary: one float per output unit of the actor
   * - multiDiscrete: one float per entry in `multiDiscreteSizes`
   */
  private readonly actionDim: number;

  /** @internal use `createPPOAgent` on the NNW instance instead of calling this directly */
  constructor(
    private readonly wasm: WasmBinding,
    private readonly actor: NeuralNetwork,
    private readonly critic: NeuralNetwork,
    config: PPOAgentConfig,
  ) {
    const cfg = { ...DEFAULTS, ...config };
    const actionSpace = resolveActionSpace(cfg.actionSpace);

    this.assertPositiveInteger(cfg.numEnvs, "numEnvs");
    this.assertPositiveInteger(cfg.rolloutSteps, "rolloutSteps");
    this.assertPositiveInteger(cfg.minibatchSize, "minibatchSize");

    this.stateDim = actor.inputDim;
    this.numEnvs = cfg.numEnvs;
    const sizes = cfg.multiDiscreteSizes ?? [];
    if (actionSpace === ActionSpace.MultiDiscrete) {
      if (sizes.length === 0 || sizes.some((size) => !Number.isInteger(size) || size < 2))
        throw new Error("multiDiscrete requires multiDiscreteSizes with integer category counts >= 2");
      if (sizes.reduce((sum, size) => sum + size, 0) !== actor.outputDim)
        throw new Error("actor.outputDim must equal the sum of multiDiscreteSizes");
    }
    this.actionDim = actionSpace === ActionSpace.Discrete
      ? 1
      : actionSpace === ActionSpace.MultiDiscrete
        ? sizes.length
        : actor.outputDim;

    const create = (sizesPtr: number, sizesCount: number) => wasm.raw._rl_ppo_create_agent(
        actor.id,
        critic.id,
        actionSpace,
        sizesPtr,
        sizesCount,
        resolveOptimizer(cfg.optimizer),
        cfg.numEnvs,
        cfg.rolloutSteps,
        cfg.learningRate,
        cfg.gamma,
        cfg.gaeLambda,
        cfg.clipRange,
        cfg.valueLossCoef,
        cfg.entropyCoef,
        cfg.maxGradNorm,
        cfg.epochs,
        cfg.minibatchSize,
      );
    if (sizes.length > 0) {
      const sizesPtr = wasm.memory.allocSizeArray(sizes);
      try {
        this.handle = create(sizesPtr, sizes.length);
      } finally {
        wasm.memory.free(sizesPtr);
      }
    } else {
      this.handle = create(0, 0);
    }

    if (!this.handle) {
      throw new Error("Failed to create PPO agent (check actor/critic dimensions and config)");
    }
  }

  /**
   * Samples actions for a batch of states.
   *
   * `actions` is a flat array of `batchSize * actionDim` floats (row-major,
   * one contiguous block per env). For discrete action spaces `actionDim`
   * is 1 (a single sampled index per env); for continuous/multiDiscrete/
   * multiBinary it equals the actor's output dimension. For multiDiscrete it
   * equals the number of configured sub-actions.
   */
  collectStep(states: number[], batchSize: number): PPOStepResult {
    this.assertAlive();
    this.assertBatchSize(batchSize);
    this.assertArrayLength(states, batchSize * this.stateDim, "collectStep.states");
    const { memory, raw } = this.wasm;

    return memory.withArrays([states], ([statesPtr]) => {
      const outActions = memory.allocOutput(batchSize * this.actionDim);
      const outLogProbs = memory.allocOutput(batchSize);
      const outValues = memory.allocOutput(batchSize);

      try {
        this.wasm.call(() =>
          raw._rl_ppo_collect_step(this.handle, statesPtr, batchSize, outActions, outLogProbs, outValues),
        );
        return {
          actions: memory.readArray(outActions, batchSize * this.actionDim),
          logProbs: memory.readArray(outLogProbs, batchSize),
          values: memory.readArray(outValues, batchSize),
        };
      } finally {
        memory.free(outActions);
        memory.free(outLogProbs);
        memory.free(outValues);
      }
    });
  }

  /** Stores one rollout step's transitions in the agent's buffer. */
  storeTransition(
    states: number[],
    batchSize: number,
    actions: number[],
    logProbs: number[],
    rewards: number[],
    terminals: number[],
    values: number[],
  ): void {
    this.assertAlive();
    this.assertBatchSize(batchSize);
    this.assertTransitionShapes({ states, actions, logProbs, rewards, terminals, values }, batchSize);
    const { memory, raw } = this.wasm;

    memory.withArrays(
      [states, actions, logProbs, rewards, terminals, values],
      ([statesPtr, actionsPtr, logProbsPtr, rewardsPtr, terminalsPtr, valuesPtr]) => {
        this.wasm.call(() =>
          raw._rl_ppo_store_transition(
            this.handle,
            statesPtr,
            batchSize,
            actionsPtr,
            logProbsPtr,
            rewardsPtr,
            terminalsPtr,
            valuesPtr,
          ),
        );
      },
    );
  }

  /** Runs a PPO update over the buffered rollout. */
  train(nextValue: number[], nextTerminal: number[]): void {
    this.assertAlive();
    if (nextValue.length !== nextTerminal.length) {
      throw new Error(
        `train() expects nextValue and nextTerminal to have the same length, got ${nextValue.length} and ${nextTerminal.length}`,
      );
    }
    this.assertArrayLength(nextValue, this.numEnvs, "train.nextValue");
    this.assertArrayLength(nextTerminal, this.numEnvs, "train.nextTerminal");
    const { memory, raw } = this.wasm;

    memory.withArrays([nextValue, nextTerminal], ([nextValuePtr, nextTerminalPtr]) => {
      this.wasm.call(() => raw._rl_ppo_train(this.handle, nextValuePtr, nextValue.length, nextTerminalPtr));
    });
  }

  /** Frees the agent. Does NOT free the actor/critic models — call `.dispose()` on those separately if desired. */
  dispose(): void {
    if (this.disposed) return;
    this.wasm.raw._rl_ppo_free_agent(this.handle);
    this.disposed = true;
  }

  private assertAlive(): void {
    if (this.disposed) throw new Error("PPOAgent used after dispose()");
  }

  private assertBatchSize(batchSize: number): void {
    this.assertPositiveInteger(batchSize, "batchSize");
    if (batchSize !== this.numEnvs) {
      throw new Error(
        `batchSize mismatch: expected ${this.numEnvs} (the numEnvs configured for this agent), got ${batchSize}`,
      );
    }
  }

  private assertPositiveInteger(value: number, label: string): void {
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`${label} must be a positive integer, got ${value}`);
    }
  }

  private assertArrayLength(values: number[], expected: number, label: string): void {
    if (values.length !== expected) {
      throw new Error(`${label} length mismatch: expected ${expected}, got ${values.length}`);
    }
  }

  private assertTransitionShapes(
    transition: {
      states: number[];
      actions: number[];
      logProbs: number[];
      rewards: number[];
      terminals: number[];
      values: number[];
    },
    batchSize: number,
  ): void {
    const expected = {
      states: batchSize * this.stateDim,
      actions: batchSize * this.actionDim,
      logProbs: batchSize,
      rewards: batchSize,
      terminals: batchSize,
      values: batchSize,
    };

    const actual = {
      states: transition.states.length,
      actions: transition.actions.length,
      logProbs: transition.logProbs.length,
      rewards: transition.rewards.length,
      terminals: transition.terminals.length,
      values: transition.values.length,
    };

    const mismatches = Object.entries(expected)
      .filter(([key, exp]) => actual[key as keyof typeof actual] !== exp)
      .map(([key, exp]) => `  - ${key}: expected ${exp}, got ${actual[key as keyof typeof actual]}`);

    if (mismatches.length > 0) {
      throw new Error(
        [
          "PPOAgent.storeTransition shape mismatch.",
          `  numEnvs=${this.numEnvs} batchSize=${batchSize} stateDim=${this.stateDim} actionDim=${this.actionDim}`,
          ...mismatches,
        ].join("\n"),
      );
    }
  }
}
