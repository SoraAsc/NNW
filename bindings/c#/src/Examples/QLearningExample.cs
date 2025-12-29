using System;
using System.Collections.Generic;
using NNW.Core.RL;

namespace Examples {
    public class QLearningExample {
        class GridWorld : IEnvironment
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
                Reset();
            }
            
            public bool Step(uint action)
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
                
                return _done;
            }
            
            public uint Reset()
            {
                _currentState = 0;
                _goalState = (_height - 1) * _width + (_width - 1);
                _done = false;
                _lastReward = 0.0f;
                return _currentState;
            }
            
            public uint GetState() => _currentState;
            public float GetReward() => _lastReward;
            public uint GetNumStates() => _width * _height;
            public uint GetNumActions() => 4;
            public bool IsDone() => _done;
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
            agent.SetPolicy(PolicyType.EpsilonGreedy, 0.1f);

            uint numEpisodes = 100;
            uint maxSteps = 100;

            Console.WriteLine("Training agent Q-Learning...\n");

            for (uint episode = 0; episode < numEpisodes; ++episode)
            {
                uint state = env.Reset();
                uint steps = 0;
                float episodeReward = 0.0f;

                while (!env.IsDone() && steps < maxSteps)
                {
                    uint action = agent.ChooseAction(state);

                    env.Step(action);

                    uint nextState = env.GetState();
                    float reward = env.GetReward();
                    bool done = env.IsDone();

                    agent.Update(state, action, reward, nextState, done);

                    episodeReward += reward;
                    state = nextState;
                    steps++;
                }

                if ((episode + 1) % 10 == 0)
                {
                    Console.WriteLine($"Episode {episode + 1}/{numEpisodes} | " +
                                    $"Reward: {episodeReward:F1} | Steps: {steps}");
                }
            }

            Console.WriteLine("\nUsing Greedy Policy...\n");
            agent.SetPolicy(PolicyType.Greedy, 0.0f);

            for (int test = 0; test < 5; ++test)
            {
                uint state = env.Reset();
                int steps = 0;
                float totalReward = 0.0f;

                while (!env.IsDone() && steps < maxSteps)
                {
                    uint action = agent.ChooseAction(state);
                    env.Step(action);
                    state = env.GetState();
                    totalReward += env.GetReward();
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
