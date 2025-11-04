import numpy as np
import ctypes
import weakref
from . import _ffi

class NNError(Exception):
    pass

def _ensure_float32_contiguous(arr):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return a

def _ptr_from_array(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

class Model:
    def __init__(self, input_dim: int):
        self._as_void_p = _ffi.lib.nn_create_model(input_dim)
        if not self._as_void_p:
            raise NNError("nn_create_model returned NULL")
        self._finalizer = weakref.finalize(self, _ffi.lib.nn_free_model, self._as_void_p)

    def add_dense(self, units: int, activation: int):
        _ffi.lib.nn_add_dense(self._as_void_p, units, activation)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_float32_contiguous(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples, x_dim = X.shape

        out_dim = _ffi.lib.nn_get_output_dim(self._as_void_p)

        out = np.zeros((n_samples, out_dim), dtype=np.float32)
        _ffi.lib.nn_predict(
            self._as_void_p,
            _ptr_from_array(X), n_samples, x_dim,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out_dim
        )
        return out

    def free(self):
        if self._finalizer.alive:
            self._finalizer()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free()

class Trainer:
    def __init__(self, model: Model, optimizer: int, loss: int, cfg=None):
        if not isinstance(model, Model):
            raise TypeError("Trainer expects a Model instance")
        if cfg is None:
            cfg = _ffi.NN_TrainerConfig()
            cfg.epochs = 1000
            cfg.batch_size = 4
            cfg.shuffle = 1
            cfg.learning_rate = 0.1
            cfg = cfg
        elif isinstance(cfg, dict):
            c = _ffi.NN_TrainerConfig()
            c.epochs = cfg.get("epochs", 1000)
            c.batch_size = cfg.get("batch_size", 4)
            c.shuffle = 1 if cfg.get("shuffle", True) else 0
            c.learning_rate = float(cfg.get("learning_rate", 0.1))
            cfg = c
        elif not isinstance(cfg, _ffi.NN_TrainerConfig):
            raise TypeError("cfg must be dict or NN_TrainerConfig")

        self._as_void_p = _ffi.lib.nn_create_trainer(model._as_void_p, optimizer, loss, ctypes.byref(cfg))
        if not self._as_void_p:
            raise NNError("nn_create_trainer returned NULL")
        self._finalizer = weakref.finalize(self, _ffi.lib.nn_free_trainer, self._as_void_p)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X = _ensure_float32_contiguous(X)
        Y = _ensure_float32_contiguous(Y)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        n_samples, x_dim = X.shape
        y_dim = Y.shape[1]

        _ffi.lib.nn_train_fit(
            self._as_void_p,
            _ptr_from_array(X), n_samples, x_dim,
            _ptr_from_array(Y), y_dim
        )

    def free(self):
        if self._finalizer.alive:
            self._finalizer()

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free()