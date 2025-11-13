using System;

class Program
{
    static void Main()
    {
        var model = NNInterop.nn_create_model((UIntPtr)3);
        NNInterop.nn_add_dense(model, (UIntPtr)4, 1); // act=1 (ReLU)
        NNInterop.nn_add_dense(model, (UIntPtr)1, 0); // act=0 (Linear)

        var cfg = new NNInterop.NN_TrainerConfig
        {
            epochs = (UIntPtr)1000,
            batch_size = (UIntPtr)2,
            shuffle = 1,
            learning_rate = 0.01f
        };

        IntPtr trainer = NNInterop.nn_create_trainer(model, 1, 0, ref cfg); // optimizer=1, loss=0

        float[] x = { 0f, 0f, 0f, 1f, 1f, 1f }; // 2 samples, 3 features
        float[] y = { 0f, 1f };                 // 2 labels

        NNInterop.nn_train_fit(trainer, x, (UIntPtr)2, (UIntPtr)3, y, (UIntPtr)1);

        float[] outBuf = new float[1];
        NNInterop.nn_predict(model, new float[] { 1f, 1f, 1f }, (UIntPtr)1, (UIntPtr)3, outBuf, (UIntPtr)1);

        Console.WriteLine("Pred: " + outBuf[0]);

        NNInterop.nn_free_trainer(trainer);
        NNInterop.nn_free_model(model);
    }
}
