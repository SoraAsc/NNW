from dataclasses import dataclass
import numpy as np
import ctypes
import weakref
from . import _ffi, enums

class NNError(Exception):
    pass

@dataclass
class TrainerConfig:
    epochs: int
    batch_size: int
    shuffle: bool
    learning_rate: float

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

    def add_dense(self, units: int, activation: enums.Activation):
        _ffi.lib.nn_add_dense(self._as_void_p, units, activation)

    def predict(self, X: np.ndarray):
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
    def __init__(self, model: Model, optimizer: enums.Optimizer, loss: enums.Loss, cfg: TrainerConfig = None):
        if not isinstance(model, Model):
            raise TypeError("Trainer expects a Model instance")
        t_config = _ffi.NN_TrainerConfig(2000, 4, 1, 0.1)
        if cfg is not None:
            t_config.epochs = cfg.epochs
            t_config.batch_size = cfg.batch_size
            t_config.shuffle = 1 if cfg.shuffle else 0
            t_config.learning_rate = cfg.learning_rate

        self._as_void_p = _ffi.lib.nn_create_trainer(model._as_void_p, optimizer, loss, ctypes.byref(t_config))
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