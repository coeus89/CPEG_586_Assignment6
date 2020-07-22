import numpy as np
from CNNEnums import *
from MyEnums import ActivationType
from Pooling import *
from ActivationFunction import *

class FeatureMap(object):
    def __init__(self,inputDataSize, poolingType, activationType, batchSize = 1):
        self.inputDataSize = inputDataSize
        self.poolingType = poolingType
        self.activationType = activationType
        self.batchSize = batchSize
        self.DeltaPool = np.zeros((batchSize,(int)(inputDataSize/2),(int)(inputDataSize/2))) # Subsampling or pooling delta
        self.DeltaCV = np.zeros((batchSize,inputDataSize,inputDataSize)) # Layer Deltas
        self.OutputPool = np.zeros((batchSize,(int)(inputDataSize/2),(int)(inputDataSize/2))) # output after pooling or subsampling
        self.ActCV = np.zeros((batchSize,inputDataSize,inputDataSize)) # Activation Function Output
        self.APrime = np.zeros((batchSize,inputDataSize,inputDataSize)) # Derivative of Activation Function
        self.Sum = np.zeros((batchSize,inputDataSize,inputDataSize)) # Sum of convolution result and bias. Before Activation Function.
        self.Bias = 0 # one bias for one feature map
        self.BiasGradient = 0
    
    def Evaluate(self,inputData,batchIndex):
        numRows = inputData.shape[0] # make sure this really gets the rows
        Res = None
        self.Sum[batchIndex] = inputData + self.Bias
        if (self.activationType == ActivationType.SIGMOID):
            self.ActCV[batchIndex], self.APrime[batchIndex] = ActivationFunction.Sigmoid(self.Sum[batchIndex])
        elif (self.activationType == ActivationType.RELU):
            self.ActCV[batchIndex], self.APrime[batchIndex] = ActivationFunction.Relu(self.Sum[batchIndex])
        elif (self.activationType == ActivationType.TANH):
            self.ActCV[batchIndex], self.APrime[batchIndex] = ActivationFunction.TanH(self.Sum[batchIndex])
        if(self.poolingType == PoolingType.AVGPOOLING):
            Res = Pooling.AvgPool(self.ActCV[batchIndex])
        self.OutputPool[batchIndex] = Res
        return Res