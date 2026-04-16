using System;
using System.Runtime.InteropServices;
using NNW.Core.NN;

namespace NNW.Interop
{
    internal static class NNInterop
    {
        #if UNITY_WEBGL && !UNITY_EDITOR
        private const string LIB_NAME = "__Internal";
        #else
        private const string LIB_NAME = "nn";
        #endif

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nn_create_model(UIntPtr input_dim);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nn_free_model(IntPtr model);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nn_add_dense(IntPtr model, UIntPtr units, Activation act);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr nn_get_input_dim(IntPtr model);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr nn_get_output_dim(IntPtr model);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nn_create_trainer(
            IntPtr model,
            Optimizer optimizer,
            Loss loss,
            ref NN_TrainerConfig cfg
        );

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nn_free_trainer(IntPtr trainer);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nn_train_fit(
            IntPtr trainer,
            float[] x, UIntPtr n_samples, UIntPtr x_dim,
            float[] y, UIntPtr y_dim
        );

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nn_predict(
            IntPtr model,
            float[] x, UIntPtr n_samples, UIntPtr x_dim,
            [Out] float[] outBuf, UIntPtr y_dim
        );

        [StructLayout(LayoutKind.Sequential)]
        public struct NN_TrainerConfig
        {
            public UIntPtr epochs;
            public UIntPtr batch_size;
            public int shuffle;
            public float learning_rate;
        }
    }
}
