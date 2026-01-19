using System;
using NNW.Interop;

namespace NNW.Core.RL
{
    public sealed class QLearningAgent : IDisposable
    {
        private NativeQLearningAgentHandle _handle;
        private readonly QTable _qtable;
        private bool _disposed;

        public uint NumStates { get; private set; }
        public uint NumActions { get; private set; }

        private QLearningAgent(NativeQLearningAgentHandle handle, uint numStates, uint numActions)
        {
            _handle = handle ?? throw new ArgumentNullException(nameof(handle));
            NumStates = numStates;
            NumActions = numActions;
            SetPolicy(PolicyType.EpsilonGreedy, 0.1f); // Default Policy

            _qtable = InitializeQTableInstance(numStates, numActions);            
        }

        private QTable InitializeQTableInstance(uint numStates, uint numActions)
        {
            IntPtr qtable_handle = RLInterop.rl_get_agent_qtable(_handle.DangerousGetHandle());
            if (qtable_handle == IntPtr.Zero) throw new InvalidOperationException("Failed to get Q-Table");
            return QTable.FromNative(qtable_handle, numStates, numActions);
        }

        public static QLearningAgent Create(uint numStates, uint numActions,
                                            float learningRate = 0.1f,
                                            float discountFactor = 0.99f)
        {
            IntPtr ptr = RLInterop.rl_create_agent(
                new UIntPtr(numStates),
                new UIntPtr(numActions),
                learningRate,
                discountFactor
            );

            if (ptr == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create native QLearningAgent.");
            return new QLearningAgent(new NativeQLearningAgentHandle(ptr), numStates, numActions);
        }

        public uint ChooseAction(uint state)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(QLearningAgent));

            UIntPtr result = RLInterop.rl_choose_agent_action(
                _handle.DangerousGetHandle(),
                new UIntPtr(state)
            );

            return checked((uint)result.ToUInt64());
        }

        public void Update(uint state, uint action, float reward, uint next_state, bool done = false)
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            RLInterop.rl_update_agent(_handle.DangerousGetHandle(), 
                                      new UIntPtr(state), 
                                      new UIntPtr(action), 
                                      reward, 
                                      new UIntPtr(next_state), 
                                      done ? 1 : 0);
        }
        
        public void SetPolicy(PolicyType type, float epsilon = 0.1f)
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            RLInterop.rl_set_agent_policy(_handle.DangerousGetHandle(), type, epsilon);
        }

        public bool SetEpsilonDecay(double start, double min, double rate, EpsilonDecayType type, bool perStep = true)
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            return RLInterop.rl_set_agent_epsilon_decay(_handle.DangerousGetHandle(), start, min, rate, (int)type, perStep ? 1 : 0);
        }

        public void UpdateEpsilonStep()
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            RLInterop.rl_update_agent_epsilon_step(_handle.DangerousGetHandle());
        }

        public void UpdateEpsilonEpisode()
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            RLInterop.rl_update_agent_epsilon_episode(_handle.DangerousGetHandle());
        }

        public double GetEpsilon()
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            return RLInterop.rl_get_agent_epsilon(_handle.DangerousGetHandle());
        }

        public void ResetEpsilon()
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            RLInterop.rl_reset_agent_epsilon(_handle.DangerousGetHandle());
        }
        
        public float GetLearningRate()
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            return RLInterop.rl_get_agent_learning_rate(_handle.DangerousGetHandle());
        }
        
        public void SetLearningRate(float lr)
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            RLInterop.rl_set_agent_learning_rate(_handle.DangerousGetHandle(), lr);
        }
        
        public float GetDiscountFactor()
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            return RLInterop.rl_get_agent_discount_factor(_handle.DangerousGetHandle());
        }
        
        public void SetDiscountFactor(float gamma)
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            RLInterop.rl_set_agent_discount_factor(_handle.DangerousGetHandle(), gamma);
        }
        
        public QTable GetQTable()
        {
            if (_disposed) throw new ObjectDisposedException("QLearningAgent");
            return _qtable;
        }

        internal NativeQLearningAgentHandle NativeHandle => _handle;

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(QLearningAgent));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _handle?.Dispose();
            _handle = null!;
            GC.SuppressFinalize(this);
        }
    }
}
