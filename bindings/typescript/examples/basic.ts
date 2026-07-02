// Minimal PPO example for CartPole-v1, using the high-level NNW API.
import { createNNW } from "../src/index.js";

// ---- CartPole-v1 environment  ----
class CartPoleEnv {
  static readonly GRAVITY = 9.8;
  static readonly MASSCART = 1.0;
  static readonly MASSPOLE = 0.1;
  static readonly TOTAL_MASS = CartPoleEnv.MASSCART + CartPoleEnv.MASSPOLE;
  static readonly HALF_POLE = 0.5;
  static readonly POLEMASS_LEN = CartPoleEnv.MASSPOLE * CartPoleEnv.HALF_POLE;
  static readonly FORCE_MAG = 10.0;
  static readonly TAU = 0.02;

  static readonly ANGLE_THRESHOLD = (12.0 * Math.PI) / 180.0;
  static readonly POS_THRESHOLD = 2.4;

  x = 0;
  x_dot = 0;
  theta = 0;
  theta_dot = 0;
  done = false;

  private rngState: number;

  constructor(seed = 0) {
    this.rngState = seed >>> 0 || 1;
    this.reset();
  }

  private nextRandom(): number {
    let t = (this.rngState += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  private uniform(lo: number, hi: number): number {
    return lo + this.nextRandom() * (hi - lo);
  }

  reset(): number[] {
    this.x = this.uniform(-0.05, 0.05);
    this.x_dot = this.uniform(-0.05, 0.05);
    this.theta = this.uniform(-0.05, 0.05);
    this.theta_dot = this.uniform(-0.05, 0.05);
    this.done = false;
    return this.state();
  }

  step(action: number): { obs: number[]; reward: number; done: boolean } {
    const force = action === 1 ? CartPoleEnv.FORCE_MAG : -CartPoleEnv.FORCE_MAG;

    const cosT = Math.cos(this.theta);
    const sinT = Math.sin(this.theta);
    const temp =
      (force + CartPoleEnv.POLEMASS_LEN * this.theta_dot * this.theta_dot * sinT) /
      CartPoleEnv.TOTAL_MASS;
    const thetaAcc =
      (CartPoleEnv.GRAVITY * sinT - cosT * temp) /
      (CartPoleEnv.HALF_POLE *
        (4.0 / 3.0 - (CartPoleEnv.MASSPOLE * cosT * cosT) / CartPoleEnv.TOTAL_MASS));
    const xAcc = temp - (CartPoleEnv.POLEMASS_LEN * thetaAcc * cosT) / CartPoleEnv.TOTAL_MASS;

    this.x += CartPoleEnv.TAU * this.x_dot;
    this.x_dot += CartPoleEnv.TAU * xAcc;
    this.theta += CartPoleEnv.TAU * this.theta_dot;
    this.theta_dot += CartPoleEnv.TAU * thetaAcc;

    this.done =
      Math.abs(this.x) > CartPoleEnv.POS_THRESHOLD ||
      Math.abs(this.theta) > CartPoleEnv.ANGLE_THRESHOLD;

    return { obs: this.state(), reward: 1.0, done: this.done };
  }

  state(): number[] {
    return [this.x, this.x_dot, this.theta, this.theta_dot];
  }
}

async function main() {
  const STATE_DIM = 4;
  const NUM_ACTIONS = 2;
  const NUM_ENVS = 8;
  const ROLLOUT_STEPS = 128;
  const TOTAL_STEPS = 200_000;

  const nnw = await createNNW();

  const actor = nnw
    .createModel(STATE_DIM)
    .addDense(64, "tanh")
    .addDense(64, "tanh")
    .addDense(NUM_ACTIONS, "linear");

  const critic = nnw
    .createModel(STATE_DIM)
    .addDense(64, "tanh")
    .addDense(64, "tanh")
    .addDense(1, "linear");

  const agent = nnw.createPPOAgent(actor, critic, {
    actionSpace: "discrete",
    optimizer: "adamw",
    numEnvs: NUM_ENVS,
    rolloutSteps: ROLLOUT_STEPS,
    minibatchSize: 64,
    // learningRate, gamma, gaeLambda, clipRange, etc. all can be overridden here if needed.
  });

  const envs = Array.from({ length: NUM_ENVS }, (_, i) => new CartPoleEnv(i * 42));

  let obs: number[] = new Array(NUM_ENVS * STATE_DIM).fill(0);
  envs.forEach((env, i) => {
    const s = env.reset();
    for (let d = 0; d < STATE_DIM; ++d) obs[i * STATE_DIM + d] = s[d];
  });

  let totalCollected = 0;
  let episodeCount = 0;
  const epRewards: number[] = [];
  const envEpReward = new Array(NUM_ENVS).fill(0);

  console.log(`Training PPO on CartPole-v1\n  envs=${NUM_ENVS}  rollout=${ROLLOUT_STEPS}  total_steps=${TOTAL_STEPS}\n`);

  while (totalCollected < TOTAL_STEPS) {
    const nextDone = new Array(NUM_ENVS).fill(0);

    for (let step = 0; step < ROLLOUT_STEPS; ++step) {
      const { actions, logProbs, values } = agent.collectStep(obs, NUM_ENVS);

      const rewards = new Array(NUM_ENVS).fill(0);
      const dones = new Array(NUM_ENVS).fill(0);
      const nextObs: number[] = new Array(NUM_ENVS * STATE_DIM).fill(0);

      envs.forEach((env, e) => {
        const action = Math.round(actions[e]);
        const { obs: nobs, reward, done } = env.step(action);

        rewards[e] = reward;
        dones[e] = done ? 1 : 0;
        envEpReward[e] += reward;

        if (done) {
          epRewards.push(envEpReward[e]);
          if (epRewards.length > 100) epRewards.shift();
          envEpReward[e] = 0;
          ++episodeCount;

          const resetObs = env.reset();
          for (let d = 0; d < STATE_DIM; ++d) nextObs[e * STATE_DIM + d] = resetObs[d];
        } else {
          for (let d = 0; d < STATE_DIM; ++d) nextObs[e * STATE_DIM + d] = nobs[d];
        }
      });

      agent.storeTransition(obs, NUM_ENVS, actions, logProbs, rewards, dones, values);
      obs = nextObs;
      totalCollected += NUM_ENVS;
    }

    const { values: bootValues } = agent.collectStep(obs, NUM_ENVS);
    agent.train(bootValues, nextDone);

    if (totalCollected % 10_000 < NUM_ENVS * ROLLOUT_STEPS) {
      const meanRew = epRewards.length ? epRewards.reduce((a, b) => a + b, 0) / epRewards.length : 0;
      console.log(`steps=${totalCollected}  episodes=${episodeCount}  mean_ep_reward(last100)=${meanRew.toFixed(2)}`);

      if (meanRew >= 195.0 && epRewards.length >= 100) {
        console.log(`\nCartPole solved in ${totalCollected} steps!`);
        break;
      }
    }
  }

  agent.dispose();
  actor.dispose();
  critic.dispose();

  console.log("\nTraining complete.");
}

main().catch((err) => {
  console.error(err);
});