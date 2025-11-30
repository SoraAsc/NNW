using System;
using System.Runtime.InteropServices;

namespace NNW.Unity
{
    public sealed class Model : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        private Model(IntPtr handle)
        {
            _handle = handle;
        }

        public static Model Create(uint inputDim)
        {
            IntPtr ptr = NNInterop.nn_create_model(new UIntPtr(inputDim));
            if (ptr == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create native model.");
            return new Model(ptr);
        }

        public void AddDense(uint units, int activation)
        {
            ThrowIfDisposed();
            NNInterop.nn_add_dense(_handle, new UIntPtr(units), activation);
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
            NNInterop.nn_predict(_handle, flatX, new UIntPtr(nSamples), new UIntPtr(xDim), outBuf, new UIntPtr(yDim));
            float[,] result = new float[nSamples, yDim];
            Buffer.BlockCopy(outBuf, 0, result, 0, outBuf.Length * sizeof(float));
            return result;
        }

        public uint GetOutputDim()
        {
            ThrowIfDisposed();
            return NNInterop.nn_get_output_dim(_handle).ToUInt32();
        }

        internal IntPtr NativeHandle => _handle;

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(Model));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_handle != IntPtr.Zero)
            {
                NNInterop.nn_free_model(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }
    }

    public sealed class Trainer : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        private Trainer(IntPtr handle)
        {
            _handle = handle;
        }

        public static Trainer Create(Model model, int optimizer, int loss, NNInterop.NN_TrainerConfig cfg)
        {
            IntPtr trainerPtr = NNInterop.nn_create_trainer(
                model.NativeHandle, optimizer, loss, ref cfg);
            if (trainerPtr == IntPtr.Zero)
                throw new InvalidOperationException("Failed to create native trainer.");
            return new Trainer(trainerPtr);
        }

        public void Fit(float[,] X, float[,] Y)
        {
            ThrowIfDisposed();
            uint nSamples = (uint)X.GetLength(0);
            uint xDim = (uint)X.GetLength(1);
            uint yDim = (uint)Y.GetLength(1);
            float[] flatX = new float[nSamples * xDim];
            float[] flatY = new float[nSamples * yDim];
            Buffer.BlockCopy(X, 0, flatX, 0, flatX.Length * sizeof(float));
            Buffer.BlockCopy(Y, 0, flatY, 0, flatY.Length * sizeof(float));
            NNInterop.nn_train_fit(_handle, flatX, new UIntPtr(nSamples), new UIntPtr(xDim), flatY, new UIntPtr(yDim));
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(Trainer));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_handle != IntPtr.Zero)
            {
                NNInterop.nn_free_trainer(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }
    }
}
