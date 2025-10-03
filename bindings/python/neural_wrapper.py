import ctypes
import os
import math

lib_path = os.path.join(os.path.dirname(__file__), '../../build/neural_wrapper.so')
lib = ctypes.CDLL(lib_path)

# class DenseLayer(ctypes.Structure):
    # _fields_ = [("weights", ctypes.c_void_p), ("biases", ctypes.c_void_p)]

lib.dense_layer_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
lib.dense_layer_create.restype = ctypes.c_void_p
# lib.dense_layer_create.restype = ctypes.POINTER(DenseLayer)

lib.dense_layer_free.argtypes = [ctypes.c_void_p]
lib.dense_layer_free.restype = None

lib.dense_info.argtypes = [ctypes.c_void_p]
lib.dense_info.restype = ctypes.c_char_p

lib.dense_detailed_info.argtypes = [ctypes.c_void_p]
lib.dense_detailed_info.restype = ctypes.c_char_p

def create_dense_layer(in_features, out_features):
    dense_layer = lib.dense_layer_create(in_features, out_features)
    return dense_layer

def free_dense_layer(layer):
    lib.dense_layer_free(layer)

# Example usage
if __name__ == "__main__":
    layer = create_dense_layer(10, 15)
    print("Layer info:", lib.dense_info(layer))

    detailed_info = lib.dense_detailed_info(layer)
    print("Detailed Layer info:", detailed_info.decode('utf-8'))

    free_dense_layer(layer)