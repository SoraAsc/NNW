using System;
using System.Collections.Generic;
using NNW.Core.RL;

namespace Examples {
    public class QLearningExample {
        class GridWorld : IEnvironment<QLearningAgent>
        {
            private uint _width, _height;
            private uint _currentState;
            private uint _goalState;
            private float _lastReward;
            private bool _done;
            
            public GridWorld(uint width = 5, uint height = 5)
            {
                _width = width;
                _height = height;
                ResetEnv();
            }
            
            public void Step(QLearningAgent _, uint action)
            {
                uint x = _currentState % _width;
                uint y = _currentState / _width;
                
                // Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
                switch (action)
                {
                    case 0: y = (y > 0) ? y - 1 : y; break;
                    case 1: y = (y < _height - 1) ? y + 1 : y; break;
                    case 2: x = (x > 0) ? x - 1 : x; break;
                    case 3: x = (x < _width - 1) ? x + 1 : x; break;
                }
                
                _currentState = y * _width + x;
                
                if (_currentState == _goalState) {
                    _lastReward = 10.0f;
                    _done = true;
                } else _lastReward = -0.1f;   
            }
            
            public void ResetEnv()
            {
                _currentState = 0;
                _goalState = (_height - 1) * _width + (_width - 1);
                _done = false;
                _lastReward = 0.0f;
            }
            
            public uint GetState(QLearningAgent _) => _currentState;
            public float GetReward(QLearningAgent _) => _lastReward;
            public uint GetNumStates() => _width * _height;
            public uint GetNumActions() => 4;
            public bool IsEnvDone() => _done;
        }
        
        public static void Run()
        {
            Console.WriteLine("=== Q-Learning ===\n");
            
            var env = new GridWorld(5, 5);

            using var agent = QLearningAgent.Create(
                env.GetNumStates(),
                env.GetNumActions(),
                0.1f,   // learning rate
                0.99f   // discount factor
            );
            agent.SetPolicy(PolicyType.EpsilonGreedy, 1.0f);
            agent.SetEpsilonDecay(1.0, 0.01, 0.0005, EpsilonDecayType.Exponential, true);
            agent.SetRewardClip(true, -10.0f, 10.0f);
            agent.SetRewardNormalization(false, 1.0f);

            uint numEpisodes = 100;
            uint maxSteps = 100;

            Console.WriteLine("Training agent Q-Learning...\n");

            for (uint episode = 0; episode < numEpisodes; ++episode)
            {
                env.ResetEnv();
                uint state = env.GetState(agent);
                uint steps = 0;
                float episodeReward = 0.0f;

                while (!env.IsEnvDone() && steps < maxSteps)
                {
                    uint action = agent.ChooseAction(state);

                    env.Step(agent, action);

                    uint nextState = env.GetState(agent);
                    float reward = env.GetReward(agent);
                    bool done = env.IsEnvDone();

                    agent.Update(state, action, reward, nextState, done);

                    // update epsilon per step
                    agent.UpdateEpsilonStep();

                    episodeReward += reward;
                    state = nextState;
                    steps++;
                }

                // If the environment didn't signal done (e.g. reached max steps), finalize
                if (!env.IsEnvDone()) agent.NotifyEpisodeEnd();

                if ((episode + 1) % 10 == 0)
                {
                    double avgReward = agent.GetAverageReward();
                    double lastReward = agent.GetLastEpisodeReward();
                    ulong episodesCount = agent.GetEpisodeCount();
                    ulong lastLen = agent.GetLastEpisodeLength();
                    double avgLen = agent.GetAverageEpisodeLength();

                    Console.WriteLine($"Episode {episode + 1}/{numEpisodes} | " +
                                    $"Reward: {episodeReward:F1} | Steps: {steps} | Epsilon: {agent.GetEpsilon():F4}");
                    Console.WriteLine($"  Telemetry: episodes={episodesCount} last_reward={lastReward:F2} avg_reward={avgReward:F2} last_len={lastLen} avg_len={avgLen:F2}");
                }
            }

            Console.WriteLine("\nTesting...\n");
            agent.SetTraining(false);

            for (int test = 0; test < 5; ++test)
            {
                env.ResetEnv();
                uint state = env.GetState(agent);
                uint steps = 0;
                float totalReward = 0.0f;

                while (!env.IsEnvDone() && steps < maxSteps)
                {
                    uint action = agent.ChooseAction(state);
                    env.Step(agent, action);
                    state = env.GetState(agent);
                    totalReward += env.GetReward(agent);
                    steps++;
                }

                Console.WriteLine($"Test {test + 1} | Reward: {totalReward:F1} | Steps: {steps}");
            }

            Console.WriteLine("\nQ-Table Params...");
            Console.WriteLine($"Number of States: {agent.NumStates}");
            Console.WriteLine($"Number of Actions: {agent.NumActions}");
            Console.WriteLine($"Learning Rate: {agent.GetLearningRate()}");
            Console.WriteLine($"Discount Factor: {agent.GetDiscountFactor()}");
        }
    }
}
