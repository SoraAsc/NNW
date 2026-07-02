export type NNActivation = 0 | 1 | 2 | 3;
export type RLActionSpaceType = 0 | 1 | 2 | 3;
export type RLOptimizerType = 0 | 1;

export interface NNWModule {
  createModel(inputDim: number): number;
  freeModel(model: number): void;
  addDense(model: number, units: number, activation: NNActivation): void;
  getInputDim(model: number): number;
  getOutputDim(model: number): number;
  createPPOAgent(
    actorModel: number,
    criticModel: number,
    actionSpace: RLActionSpaceType,
    optimizerType: RLOptimizerType,
    numEnvs: number,
    rolloutSteps: number,
    learningRate: number,
    gamma: number,
    gaeLambda: number,
    clipRange: number,
    valueLossCoef: number,
    entropyCoef: number,
    maxGradNorm: number,
    epochs: number,
    minibatchSize: number,
  ): number;
  freePPOAgent(agent: number): void;
  ppoCollectStep(
    agent: number,
    states: number[],
    batchSize: number,
  ): { actions: number[]; logProbs: number[]; values: number[] };
  ppoStoreTransition(
    agent: number,
    states: number[],
    batchSize: number,
    actions: number[],
    logProbs: number[],
    rewards: number[],
    terminals: number[],
    values: number[],
  ): void;
  ppoTrain(agent: number, nextValue: number[], nextTerminal: number[]): void;
}

interface EmscriptenModuleLike {
  _malloc: (size: number) => number;
  _free: (ptr: number) => void;
  _nn_create_model: (inputDim: number) => number;
  _nn_free_model: (model: number) => void;
  _nn_add_dense: (model: number, units: number, activation: number) => void;
  _nn_get_input_dim: (model: number) => number;
  _nn_get_output_dim: (model: number) => number;
  _rl_ppo_create_agent: (...args: number[]) => number;
  _rl_ppo_free_agent: (agent: number) => void;
  _rl_ppo_collect_step: (agent: number, states: number, batchSize: number, outActions: number, outLogProbs: number, outValues: number) => void;
  _rl_ppo_store_transition: (agent: number, states: number, batchSize: number, actions: number, logProbs: number, rewards: number, terminals: number, values: number) => void;
  _rl_ppo_train: (agent: number, nextValue: number, batchSize: number, nextTerminal: number) => void;
  HEAPF32: Float32Array;
  HEAPU8: Uint8Array;
}

export async function createNNWModule(modulePath = "../libs/nn.mjs"): Promise<NNWModule> {
  const moduleUrl = new URL("../libs/nn.mjs", import.meta.url);
  const modFactory = (await import(moduleUrl.href)).default as (options?: Record<string, unknown>) => Promise<EmscriptenModuleLike>;
  const mod = await modFactory({
    locateFile: (name: string) => {
      if (name.endsWith(".wasm")) return new URL("../libs/" + name, import.meta.url).href;
      return name;
    },
  });

  const allocateArray = (values: number[]): { ptr: number; length: number } => {
    const length = values.length;
    const bytes = length * Float32Array.BYTES_PER_ELEMENT;
    const ptr = mod._malloc(bytes);
    const view = new Float32Array(mod.HEAPF32.buffer, ptr, length);
    view.set(values);
    return { ptr, length };
  };

  const readArray = (ptr: number, length: number): number[] => {
    if (ptr === 0) return [];
    return Array.from(new Float32Array(mod.HEAPF32.buffer, ptr, length));
  };

  const freeArray = (ptr: number): void => {
    if (ptr !== 0) mod._free(ptr);
  };

  return {
    createModel(inputDim: number) {
      return mod._nn_create_model(inputDim);
    },
    freeModel(model: number) {
      mod._nn_free_model(model);
    },
    addDense(model: number, units: number, activation: NNActivation) {
      mod._nn_add_dense(model, units, activation);
    },
    getInputDim(model: number) {
      return mod._nn_get_input_dim(model);
    },
    getOutputDim(model: number) {
      return mod._nn_get_output_dim(model);
    },
    createPPOAgent(actorModel, criticModel, actionSpace, optimizerType, numEnvs, rolloutSteps, learningRate, gamma, gaeLambda, clipRange, valueLossCoef, entropyCoef, maxGradNorm, epochs, minibatchSize) {
      return mod._rl_ppo_create_agent(
        actorModel,
        criticModel,
        actionSpace,
        optimizerType,
        numEnvs,
        rolloutSteps,
        learningRate,
        gamma,
        gaeLambda,
        clipRange,
        valueLossCoef,
        entropyCoef,
        maxGradNorm,
        epochs,
        minibatchSize,
      );
    },
    freePPOAgent(agent: number) {
      mod._rl_ppo_free_agent(agent);
    },
    ppoCollectStep(agent: number, states: number[], batchSize: number) {
      const statesAlloc = allocateArray(states);
      const outActions = mod._malloc(batchSize * Float32Array.BYTES_PER_ELEMENT);
      const outLogProbs = mod._malloc(batchSize * Float32Array.BYTES_PER_ELEMENT);
      const outValues = mod._malloc(batchSize * Float32Array.BYTES_PER_ELEMENT);

      mod._rl_ppo_collect_step(agent, statesAlloc.ptr, batchSize, outActions, outLogProbs, outValues);

      const actions = readArray(outActions, batchSize);
      const logProbs = readArray(outLogProbs, batchSize);
      const values = readArray(outValues, batchSize);

      freeArray(statesAlloc.ptr);
      mod._free(outActions);
      mod._free(outLogProbs);
      mod._free(outValues);

      return { actions, logProbs, values };
    },
    ppoStoreTransition(agent: number, states: number[], batchSize: number, actions: number[], logProbs: number[], rewards: number[], terminals: number[], values: number[]) {
      const statesAlloc = allocateArray(states);
      const actionsAlloc = allocateArray(actions);
      const logProbsAlloc = allocateArray(logProbs);
      const rewardsAlloc = allocateArray(rewards);
      const terminalsAlloc = allocateArray(terminals);
      const valuesAlloc = allocateArray(values);

      mod._rl_ppo_store_transition(agent, statesAlloc.ptr, batchSize, actionsAlloc.ptr, logProbsAlloc.ptr, rewardsAlloc.ptr, terminalsAlloc.ptr, valuesAlloc.ptr);

      freeArray(statesAlloc.ptr);
      freeArray(actionsAlloc.ptr);
      freeArray(logProbsAlloc.ptr);
      freeArray(rewardsAlloc.ptr);
      freeArray(terminalsAlloc.ptr);
      freeArray(valuesAlloc.ptr);
    },
    ppoTrain(agent: number, nextValue: number[], nextTerminal: number[]) {
      const nextValueAlloc = allocateArray(nextValue);
      const nextTerminalAlloc = allocateArray(nextTerminal);

      mod._rl_ppo_train(agent, nextValueAlloc.ptr, nextValue.length, nextTerminalAlloc.ptr);

      freeArray(nextValueAlloc.ptr);
      freeArray(nextTerminalAlloc.ptr);
    },
  };
}
