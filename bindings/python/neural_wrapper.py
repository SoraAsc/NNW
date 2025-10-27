import ctypes
import os
import numpy as np

LIB_PATH = os.path.join(os.path.dirname(__file__), '../../build/nn.so')

# Load shared library
try:
    lib = ctypes.CDLL(LIB_PATH)
except OSError as e:
    raise OSError(f"Could not load shared library '{LIB_PATH}': {e}")

class NN_TrainerConfig(ctypes.Structure):
    _fields_ = [
        ("epochs", ctypes.c_size_t),
        ("batch_size", ctypes.c_size_t),
        ("shuffle", ctypes.c_int),
        ("learning_rate", ctypes.c_float),
    ]

lib.nn_create_model.restype = ctypes.c_void_p
lib.nn_create_model.argtypes = [ctypes.c_size_t]

lib.nn_add_dense.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

lib.nn_create_trainer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(NN_TrainerConfig)]
lib.nn_create_trainer.restype = ctypes.c_void_p


lib.nn_train_fit.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
]

lib.nn_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
]

lib.nn_free_model.argtypes = [ctypes.c_void_p]
lib.nn_free_trainer.argtypes = [ctypes.c_void_p]

# --- Exemplo de uso ---
if __name__ == "__main__":
    # Criar modelo
    model = lib.nn_create_model(2)
    lib.nn_add_dense(model, 4, 1)  # RELU
    lib.nn_add_dense(model, 1, 2)  # SIGMOID

    # Configurar treinador
    cfg = NN_TrainerConfig(epochs=300, batch_size=4, shuffle=1, learning_rate=0.01)
    trainer = lib.nn_create_trainer(model, 1, 0, ctypes.byref(cfg))  # ADAMW, MSE

    # Dataset XOR
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

    lib.nn_train_fit(
        trainer,
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), X.shape[0], X.shape[1],
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), Y.shape[1]
    )

    # Previsão
    out = np.zeros((4,1), dtype=np.float32)
    lib.nn_predict(
        model,
        X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), X.shape[0], X.shape[1],
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.shape[1]
    )
    print("Predições:", out)

    # Liberar memória
    lib.nn_free_trainer(trainer)
    lib.nn_free_model(model)