using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

public static class NNInterop
{
    private const string LIB_NAME = "nn";

    static NNInterop()
    {
        NativeLibrary.SetDllImportResolver(typeof(NNInterop).Assembly, ResolveNativeLibrary);
    }

    private static IntPtr ResolveNativeLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != LIB_NAME) 
            return IntPtr.Zero;

        string asmDir = Path.GetDirectoryName(assembly.Location) ?? AppContext.BaseDirectory;

        string rel = Path.Combine(asmDir, "..", "..", "..", "build",
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nn.dll" : "nn.so");

        string candidate = Path.GetFullPath(rel);

        if (File.Exists(candidate)) return NativeLibrary.Load(candidate);
        
        // fallback
        string alt = Path.Combine(AppContext.BaseDirectory,
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nn.dll" : "nn.so");

        if (File.Exists(alt)) return NativeLibrary.Load(alt);

        return IntPtr.Zero; // not found 
    }

    [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr nn_create_model(UIntPtr input_dim);

    [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
    public static extern void nn_free_model(IntPtr model);

    [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
    public static extern void nn_add_dense(IntPtr model, UIntPtr units, int act);

    [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
    public static extern UIntPtr nn_get_input_dim(IntPtr model);

    [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
    public static extern UIntPtr nn_get_output_dim(IntPtr model);

    [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr nn_create_trainer(
        IntPtr model,
        int optimizer,
        int loss,
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
