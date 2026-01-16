using System;
using NNW.Interop;

namespace NNW.Core.RL
{
    public sealed class QTable : IDisposable
    {
        private NativeQTableHandle _handle;
        private bool _disposed;

        public uint NumStates { get; private set; }
        public uint NumActions { get; private set; }

        private QTable(NativeQTableHandle handle, uint numStates, uint numActions)
        {
            _handle = handle ?? throw new ArgumentNullException(nameof(handle));
            NumStates = numStates;
            NumActions = numActions;
        }

        internal static QTable FromNative(IntPtr ptr, uint statesNum, uint actionsNum)
        {
            if (ptr == IntPtr.Zero) throw new ArgumentNullException(nameof(ptr));
            return new QTable(new NativeQTableHandle(ptr), statesNum, actionsNum);
        }

        public float Get(uint state, uint action)
        {
            if (_disposed) throw new ObjectDisposedException("QTable");
            return RLInterop.rl_get_qtable(_handle.DangerousGetHandle(), new UIntPtr(state), new UIntPtr(action));
        }
        
        public void Set(uint state, uint action, float value)
        {
            if (_disposed) throw new ObjectDisposedException("QTable");
            RLInterop.rl_set_qtable(_handle.DangerousGetHandle(), new UIntPtr(state), new UIntPtr(action), value);
        }

        internal NativeQTableHandle NativeHandle => _handle;

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(QTable));
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
