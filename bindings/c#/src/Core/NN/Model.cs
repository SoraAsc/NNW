using System;
using NNW.Interop;

namespace NNW.Core.NN
{
    public sealed class Model : IDisposable
    {
        private NativeModelHandle _handle;
        private bool _disposed;

        private Model(NativeModelHandle handle)
        {
            _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        }

        public static Model Create(uint inputDim)
        {
            IntPtr ptr = NNInterop.nn_create_model(new UIntPtr(inputDim));
            if (ptr == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create native model.");

            return new Model(new NativeModelHandle(ptr));
        }

        public void AddDense(uint units, Activation activation)
        {
            ThrowIfDisposed();
            bool added = false;
            _handle.DangerousAddRef(ref added);
            try
            {
                NNInterop.nn_add_dense(_handle.DangerousGetHandle(), new UIntPtr(units), (int)activation);
            }
            finally
            {
                if (added) _handle.DangerousRelease();
            }
        }

        public float[,] Predict(float[,] X)
        {
            ThrowIfDisposed();

            uint nSamples = (uint)X.GetLength(0);
            uint xDim = (uint)X.GetLength(1);
            uint yDim = GetOutputDim();

            float[] flatX = new float[nSamples * xDim];
            Buffer.BlockCopy(X, 0, flatX, 0, flatX.Length * sizeof(float));

            float[] outBuf = new float[nSamples * yDim];

            bool added = false;
            _handle.DangerousAddRef(ref added);
            try
            {
                NNInterop.nn_predict(_handle.DangerousGetHandle(),
                    flatX, new UIntPtr(nSamples), new UIntPtr(xDim),
                    outBuf, new UIntPtr(yDim));
            }
            finally
            {
                if (added) _handle.DangerousRelease();
            }

            float[,] result = new float[nSamples, yDim];
            Buffer.BlockCopy(outBuf, 0, result, 0, outBuf.Length * sizeof(float));
            return result;
        }

        public uint GetOutputDim()
        {
            ThrowIfDisposed();
            bool added = false;
            _handle.DangerousAddRef(ref added);
            try
            {
                return NNInterop.nn_get_output_dim(_handle.DangerousGetHandle()).ToUInt32();
            }
            finally
            {
                if (added) _handle.DangerousRelease();
            }
        }

        internal NativeModelHandle NativeHandle => _handle;

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(Model));
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
