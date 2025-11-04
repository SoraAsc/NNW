import ctypes
import os
import sys

LIB_REL = os.path.join(os.path.dirname(__file__), '../../../build/nn.so')
if sys.platform == "win32":
    LIB_REL = os.path.join(os.path.dirname(__file__), '../../../build/nn.dll')

LIB_PATH = os.path.abspath(LIB_REL)

try:
    lib = ctypes.CDLL(LIB_PATH)
except OSError as e:
    raise OSError(f"Could not load shared library '{LIB_PATH}': {e}")

# Structures
class NN_TrainerConfig(ctypes.Structure):
    _fields_ = [
        ("epochs", ctypes.c_size_t),
        ("batch_size", ctypes.c_size_t),
        ("shuffle", ctypes.c_int),
        ("learning_rate", ctypes.c_float),
    ]

# Function prototypes
lib.nn_create_model.restype = ctypes.c_void_p
lib.nn_create_model.argtypes = [ctypes.c_size_t]

lib.nn_add_dense.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
lib.nn_add_dense.restype = None


lib.nn_get_input_dim.argtypes = [ctypes.c_void_p]
lib.nn_get_input_dim.restype = ctypes.c_size_t
lib.nn_get_output_dim.argtypes = [ctypes.c_void_p]
lib.nn_get_output_dim.restype = ctypes.c_size_t

lib.nn_create_trainer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(NN_TrainerConfig)]
lib.nn_create_trainer.restype = ctypes.c_void_p

lib.nn_train_fit.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
]
lib.nn_train_fit.restype = None

lib.nn_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
]
lib.nn_predict.restype = None

lib.nn_free_model.argtypes = [ctypes.c_void_p]
lib.nn_free_model.restype = None

lib.nn_free_trainer.argtypes = [ctypes.c_void_p]
lib.nn_free_trainer.restype = None