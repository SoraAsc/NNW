using System;
using Microsoft.Win32.SafeHandles;

namespace NNWrapper
{
    public sealed class NativeModelHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        public NativeModelHandle() : base(true) { }
        public NativeModelHandle(IntPtr handle) : base(true) { SetHandle(handle); }

        protected override bool ReleaseHandle()
        {
            NNInterop.nn_free_model(handle);
            return true;
        }
    }

    public sealed class NativeTrainerHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        public NativeTrainerHandle() : base(true) { }
        public NativeTrainerHandle(IntPtr handle) : base(true) { SetHandle(handle); }

        protected override bool ReleaseHandle()
        {
            NNInterop.nn_free_trainer(handle);
            return true;
        }
    }
}
