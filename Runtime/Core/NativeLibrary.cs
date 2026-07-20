using NNW.Interop;

namespace NNW.Core
{
    public static class NativeLibrary
    {
        public static bool IsReady()
        {
            return NNWInterop.nn_is_ready() != 0;
        }

        public static void Initialize()
        {
            NNWInterop.nn_init_module();
        }
    }
}
