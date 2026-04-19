using System;
using System.Runtime.InteropServices;

namespace NNW.Interop
{
    internal static class NNWInterop
    {
        #if UNITY_WEBGL && !UNITY_EDITOR
        private const string LIB_NAME = "__Internal";
        #else
        private const string LIB_NAME = "nn";
        #endif

        #if UNITY_WEBGL && !UNITY_EDITOR
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nn_is_ready();
        #else
        public static int nn_is_ready() { return 1; }
        #endif

        #if UNITY_WEBGL && !UNITY_EDITOR
        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nn_init_module();
        #else
        public static void nn_init_module() { }
        #endif
    }
}
