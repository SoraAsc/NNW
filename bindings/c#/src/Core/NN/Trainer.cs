using System;
using NNW.Interop;

namespace NNW.Core.NN
{
    public sealed class Trainer : IDisposable
    {
        private NativeTrainerHandle _handle;
        private bool _disposed;

        private Trainer(NativeTrainerHandle handle)
        {
            _handle = handle ?? throw new ArgumentNullException(nameof(handle));
        }

        public static Trainer Create(Model model, Optimizer opt, Loss loss, TrainerConfig cfg)
        {
            var nativeCfg = cfg.ToNative();

            bool added = false;
            model.NativeHandle.DangerousAddRef(ref added);
            try
            {
                IntPtr trainerPtr = NNInterop.nn_create_trainer(
                    model.NativeHandle.DangerousGetHandle(),
                    (int)opt, (int)loss, ref nativeCfg);

                if (trainerPtr == IntPtr.Zero)
                    throw new InvalidOperationException("Failed to create native trainer.");

                return new Trainer(new NativeTrainerHandle(trainerPtr));
            }
            finally
            {
                if (added) model.NativeHandle.DangerousRelease();
            }
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

            Fit(flatX, nSamples, xDim, flatY, yDim);
        }

        private void Fit(float[] x, uint nSamples, uint xDim, float[] y, uint yDim)
        {
            bool added = false;
            _handle.DangerousAddRef(ref added);
            try
            {
                NNInterop.nn_train_fit(
                    _handle.DangerousGetHandle(),
                    x, new UIntPtr(nSamples), new UIntPtr(xDim),
                    y, new UIntPtr(yDim)
                );
            }
            finally
            {
                if (added) _handle.DangerousRelease();
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(Trainer));
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
