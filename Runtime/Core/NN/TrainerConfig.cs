using System;
using NNW.Interop;

namespace NNW.Core.NN
{
    public sealed class TrainerConfig
    {
        public ulong Epochs { get; set; } = 500;
        public ulong BatchSize { get; set; } = 4;
        public bool Shuffle { get; set; } = true;
        public float LearningRate { get; set; } = 0.1f;

        internal NNInterop.NN_TrainerConfig ToNative()
        {
            return new NNInterop.NN_TrainerConfig
            {
                epochs = new UIntPtr(Epochs),
                batch_size = new UIntPtr(BatchSize),
                shuffle = Shuffle ? 1 : 0,
                learning_rate = LearningRate
            };
        }
    }
}
