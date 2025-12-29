using System;
using Microsoft.Win32.SafeHandles;
using NNW.Interop;

namespace NNW.Core.RL
{
    public sealed class NativeQTableHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        public NativeQTableHandle() : base(true) { }
        public NativeQTableHandle(IntPtr handle) : base(true) { SetHandle(handle); }

        protected override bool ReleaseHandle()
        {
            // RLInterop.rl_free_qtable(handle); // now is not necessary, because QLearningAgent own the QTable
            return true;
        }
    }

    public sealed class NativeQLearningAgentHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        public NativeQLearningAgentHandle() : base(true) { }
        public NativeQLearningAgentHandle(IntPtr handle) : base(true) { SetHandle(handle); }

        protected override bool ReleaseHandle()
        {
            RLInterop.rl_free_agent(handle);
            return true;
        }
    }
}
