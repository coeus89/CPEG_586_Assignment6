from enum import Enum
class GradDescType(Enum):
    Stochastic = 0
    MiniBatch = 1
    Batch = 2

class ActivationType(Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3
    SOFTMAX = 4

class LROptimizerType(Enum):
    NONE = 1
    ADAM = 2

class BatchNormMode(Enum):
    TRAIN = 1
    TEST = 2

class BatchNormEnabled(Enum):
    Enabled = 1
    Disabled = 0