import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), '../../build/neural_wrapper.so')
lib = ctypes.CDLL(lib_path)

lib.tensor_create.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
lib.tensor_create.restype = ctypes.c_void_p

lib.tensor_free.argtypes = [ctypes.c_void_p]
lib.tensor_free.restype = None

lib.tensor_data.argtypes = [ctypes.c_void_p]
lib.tensor_data.restype = ctypes.POINTER(ctypes.c_float)

lib.tensor_shape.argtypes = [ctypes.c_void_p]
lib.tensor_shape.restype = ctypes.POINTER(ctypes.c_size_t)

def create_tensor(shape):
    shape_array = (ctypes.c_size_t * len(shape))(*shape)
    tensor_ptr = lib.tensor_create(shape_array, len(shape))
    return tensor_ptr

def free_tensor(tensor_ptr):
    lib.tensor_free(tensor_ptr)

def get_tensor_data(tensor_ptr, size):
    data_ptr = lib.tensor_data(tensor_ptr)
    data_ptr[3] = 42.0  # Example modification
    return [data_ptr[i] for i in range(size)]

# Example usage
if __name__ == "__main__":
    shape = (2, 3)
    tensor = create_tensor(shape)
    data = get_tensor_data(tensor, 6)
    print("Tensor data:", data)

    shape_ptr = lib.tensor_shape(tensor)
    shape = [shape_ptr[i] for i in range(2)]  # Assuming 2D tensor
    print("Tensor shape:", shape)

    free_tensor(tensor)