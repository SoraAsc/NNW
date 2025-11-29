using NNWrapper;

namespace Examples
{
    public static class AndExample
    {
        public static void Run()
        {
            Console.WriteLine("=== AND example ===");

            using var model = Model.Create(2);
            model.AddDense(1, Activation.SIGMOID);

            var cfg = new TrainerConfig
            {
                Epochs = 3000,
                BatchSize = 4,
                LearningRate = 0.01f
            };

            using var trainer = Trainer.Create(model, Optimizer.ADAMW, Loss.MSE, cfg);

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
                {0},
                {0},
                {1}
            };

            trainer.Fit(X, Y);

            var preds = model.Predict(X);
            Console.WriteLine("Predictions:");
            for (int i = 0; i < preds.GetLength(0); i++)
                Console.WriteLine($"{X[i,0]} AND {X[i,1]} = {preds[i,0]:F4}");
        }
    }
}
