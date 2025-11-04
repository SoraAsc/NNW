import numpy as np
from nnw import Model, Trainer, Activation, Optimizer, Loss

def main():
    with Model(2) as model:
        model.add_dense(4, Activation.RELU)
        model.add_dense(1, Activation.SIGMOID)

        cfg = {"epochs": 3000, "batch_size": 4, "shuffle": True, "learning_rate": 0.01}
        with Trainer(model, Optimizer.ADAMW, Loss.MSE, cfg) as trainer:
            X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
            Y = np.array([[0],[1],[1],[0]], dtype=np.float32)
            trainer.fit(X, Y)

            preds = model.predict(X)
            print("Predictions:\n", preds)

if __name__ == "__main__":
    main()