import ctypes
import os
import sys
import importlib.resources as resources

def _load_native():
    libname = "nn.dll" if sys.platform == "win32" else "nn.so"
    try:
        lib_path = resources.files("nnw").joinpath("libs", libname)
        if lib_path.is_file():
            return ctypes.CDLL(str(lib_path))
    except Exception:
        pass
    here = os.path.dirname(__file__)
    local_lib_path = os.path.abspath(os.path.join(here, "..", "..", "..", "build", libname))
    if os.path.exists(local_lib_path):
        return ctypes.CDLL(local_lib_path)
    raise OSError(f"Could not find native library {libname}.")

lib = _load_native()

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