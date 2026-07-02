import type { WasmBinding } from "../core/wasm.js";
import type { NeuralNetwork } from "../nn/index.js";
import {
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

  /** @internal use `createPPOAgent` on the NNW instance instead of calling this directly */
  constructor(
    private readonly wasm: WasmBinding,
    private readonly actor: NeuralNetwork,
    private readonly critic: NeuralNetwork,
    config: PPOAgentConfig,
  ) {
    const cfg = { ...DEFAULTS, ...config };

    this.handle = wasm.raw._rl_ppo_create_agent(
      actor.id,
      critic.id,
      resolveActionSpace(cfg.actionSpace),
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

    if (!this.handle) {
      throw new Error("Failed to create PPO agent (check actor/critic dimensions and config)");
    }
  }

  /** Samples actions for a batch of states. */
  collectStep(states: number[], batchSize: number): PPOStepResult {
    this.assertAlive();
    const { memory, raw } = this.wasm;

    return memory.withArrays([states], ([statesPtr]) => {
      const outActions = memory.allocOutput(batchSize);
      const outLogProbs = memory.allocOutput(batchSize);
      const outValues = memory.allocOutput(batchSize);

      try {
        raw._rl_ppo_collect_step(this.handle, statesPtr, batchSize, outActions, outLogProbs, outValues);
        return {
          actions: memory.readArray(outActions, batchSize),
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
    const { memory, raw } = this.wasm;

    memory.withArrays(
      [states, actions, logProbs, rewards, terminals, values],
      ([statesPtr, actionsPtr, logProbsPtr, rewardsPtr, terminalsPtr, valuesPtr]) => {
        raw._rl_ppo_store_transition(
          this.handle,
          statesPtr,
          batchSize,
          actionsPtr,
          logProbsPtr,
          rewardsPtr,
          terminalsPtr,
          valuesPtr,
        );
      },
    );
  }

  /** Runs a PPO update over the buffered rollout. */
  train(nextValue: number[], nextTerminal: number[]): void {
    this.assertAlive();
    const { memory, raw } = this.wasm;

    memory.withArrays([nextValue, nextTerminal], ([nextValuePtr, nextTerminalPtr]) => {
      raw._rl_ppo_train(this.handle, nextValuePtr, nextValue.length, nextTerminalPtr);
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
}