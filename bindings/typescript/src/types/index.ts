export enum Activation {
  Linear = 0,
  ReLU = 1,
  Sigmoid = 2,
  Tanh = 3,
}

export type ActivationLike = Activation | "linear" | "sigmoid" | "relu" | "tanh";

export enum ActionSpace {
  Discrete = 0,
  Continuous = 1,
  MultiDiscrete = 2,
  MultiBinary = 3,
}

export type ActionSpaceLike = ActionSpace | "discrete" | "continuous" | "multiDiscrete" | "multiBinary";

export enum Optimizer {
  SGD = 0,
  AdamW = 1,
}

export type OptimizerLike = Optimizer | "sgd" | "adamw";

export interface PPOAgentConfig {
  actionSpace: ActionSpaceLike;
  /** @default "adamw" */
  optimizer?: OptimizerLike;
  numEnvs: number;
  rolloutSteps: number;
  /** @default 3e-4 */
  learningRate?: number;
  /** @default 0.99 */
  gamma?: number;
  /** @default 0.95 */
  gaeLambda?: number;
  /** @default 0.2 */
  clipRange?: number;
  /** @default 0.5 */
  valueLossCoef?: number;
  /** @default 0.01 */
  entropyCoef?: number;
  /** @default 0.5 */
  maxGradNorm?: number;
  /** @default 4 */
  epochs?: number;
  minibatchSize: number;
}

export interface PPOStepResult {
  actions: number[];
  logProbs: number[];
  values: number[];
}

const ACTIVATION_MAP: Record<string, Activation> = {
  linear: Activation.Linear,
  sigmoid: Activation.Sigmoid,
  relu: Activation.ReLU,
  tanh: Activation.Tanh,
};

const ACTION_SPACE_MAP: Record<string, ActionSpace> = {
  discrete: ActionSpace.Discrete,
  continuous: ActionSpace.Continuous,
  multidiscrete: ActionSpace.MultiDiscrete,
  multibinary: ActionSpace.MultiBinary,
};

const OPTIMIZER_MAP: Record<string, Optimizer> = {
  sgd: Optimizer.SGD,
  adamw: Optimizer.AdamW,
};

function resolve<T extends number>(map: Record<string, T>, value: T | string, kind: string): T {
  if (typeof value === "number") return value;
  const key = value.toLowerCase();
  const resolved = map[key];
  if (resolved === undefined)
    throw new Error(`Unknown ${kind}: "${value}". Valid options: ${Object.keys(map).join(", ")}`);
  return resolved;
}

export function resolveActivation(value: ActivationLike): Activation {
  return resolve(ACTIVATION_MAP, value as Activation | string, "activation");
}

export function resolveActionSpace(value: ActionSpaceLike): ActionSpace {
  return resolve(ACTION_SPACE_MAP, value as ActionSpace | string, "action space");
}

export function resolveOptimizer(value: OptimizerLike): Optimizer {
  return resolve(OPTIMIZER_MAP, value as Optimizer | string, "optimizer");
}