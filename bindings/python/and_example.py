import numpy as np
from nnw import Model, Trainer, Activation, Optimizer, Loss, TrainerConfig

def main():
    with Model(2) as model:
        model.add_dense(1, Activation.SIGMOID)

        cfg = TrainerConfig(3000, 4, True, 0.01)
        with Trainer(model, Optimizer.ADAMW, Loss.MSE, cfg) as trainer:
            X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
            Y = np.array([[0],[0],[0],[1]], dtype=np.float32)
            trainer.fit(X, Y)

            preds = model.predict(X)
            print("Predictions:\n", np.round(preds, 2))

if __name__ == "__main__":
    main()