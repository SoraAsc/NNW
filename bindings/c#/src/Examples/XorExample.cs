using NNWrapper;

namespace Examples
{
    public static class XorExample
    {
        public static void Run()
        {
            Console.WriteLine("=== XOR example ===");

            using var model = Model.Create(2);
            model.AddDense(4, Activation.RELU); // hidden
            model.AddDense(1, Activation.SIGMOID); // output

            var cfg = new TrainerConfig
            {
                Epochs = 3000,
                BatchSize = 4,
                LearningRate = 0.1f
            };

            using var trainer = Trainer.Create(model, Optimizer.SGD, Loss.MSE, cfg);

            float[,] X =
            {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
            };

            float[,] Y =
            {
                {0},
                {1},
                {1},
                {0}
            };

            trainer.Fit(X, Y);

            var preds = model.Predict(X);
            Console.WriteLine("Predictions:");
            for (int i = 0; i < preds.GetLength(0); i++)
                Console.WriteLine($"{X[i,0]} XOR {X[i,1]} = {preds[i,0]:F4}");
        }
    }
}
