using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace NNW.Interop
{
    internal static class NativeNNResolver
    {
        private const string LIB_NAME = "nn";
        private static bool _initialized;

        public static void EnsureLoaded()
        {
            if (_initialized) return;

            NativeLibrary.SetDllImportResolver(
                Assembly.GetExecutingAssembly(),
                Resolve
            );

            _initialized = true;
        }

        private static IntPtr Resolve(
            string libraryName,
            Assembly assembly,
            DllImportSearchPath? searchPath
        )
        {
            if (libraryName != LIB_NAME)
                return IntPtr.Zero;

            string fileName =
                RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                    ? "nn.dll"
                    : "nn.so";

            string baseDir = AppContext.BaseDirectory;

            // 1) executable dir
            string direct = Path.Combine(baseDir, fileName);
            if (File.Exists(direct))
                return NativeLibrary.Load(direct);

            // 2) runtimes/{rid}/native
            string rid =
                RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                    ? "win-x64"
                    : "linux-x64";

            string runtimePath = Path.Combine(
                baseDir,
                "runtimes",
                rid,
                "native",
                fileName
            );

            if (File.Exists(runtimePath))
                return NativeLibrary.Load(runtimePath);

            return IntPtr.Zero;
        }
    }
}
