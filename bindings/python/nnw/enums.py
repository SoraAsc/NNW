from enum import IntEnum

class Activation(IntEnum):
    LINEAR = 0
    RELU = 1
    SIGMOID = 2
    TANH = 3

class Optimizer(IntEnum):
    SGD = 0
    ADAMW = 1

class Loss(IntEnum):
    MSE = 0
