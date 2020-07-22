from enum import Enum


class PoolingType(Enum):
    NONE = 0
    MAXPOOLING = 1
    AVGPOOLING = 2

# class ActivationType(Enum):
#     SIGMOID = 1
#     TANH = 2
#     RELU = 3
#     SOFTMAX = 4