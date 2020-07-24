from FeatureMap import FeatureMap
import numpy as np
from CNNEnums import *
from Pooling import *
from scipy.signal import convolve2d

class CNNLayer(object):
    def __init__(self,numFeatureMaps,numPrevLayerFeatureMaps,inputSize,kernelSize,poolingType,activationType,batchSize=1):
        self.batchSize = batchSize
        self.ConvolSums = np.empty((batchSize,numFeatureMaps),dtype=object)
        self.kernelSize = kernelSize
        self.numFeatureMaps = numFeatureMaps
        self.numPrevLayerFeatureMaps = numPrevLayerFeatureMaps
        self.convOutputSize = np.array([inputSize[0] - kernelSize + 1, inputSize[1] - kernelSize + 1])
        self.ConvolResults = np.zeros((batchSize,numPrevLayerFeatureMaps,numFeatureMaps,self.convOutputSize[0],self.convOutputSize[1]))
        self.poolOutputSize = np.array([(int)(self.convOutputSize[0] / 2),(int)(self.convOutputSize[1] / 2)])
        for i in range(0,batchSize):
            for j in range(0,numFeatureMaps):
                self.ConvolSums[i,j] = np.zeros((self.convOutputSize[0],self.convOutputSize[1]))
        initRange = (numFeatureMaps/((numPrevLayerFeatureMaps + numFeatureMaps)*(kernelSize**2)))**0.5
        #self.Kernels = np.empty((numPrevLayerFeatureMaps,numFeatureMaps),dtype=object)
        self.Kernels = np.zeros((numPrevLayerFeatureMaps,numFeatureMaps,kernelSize,kernelSize))
        self.KernelGrads = np.zeros((numPrevLayerFeatureMaps,numFeatureMaps,kernelSize,kernelSize)) 
        self.KernelGradsSiamese = np.zeros((numPrevLayerFeatureMaps,numFeatureMaps,kernelSize,kernelSize)) # for siamese

        self.InitializeKernels(initRange)
        self.featureMapList = []
        for i in range(0,numFeatureMaps):
            self.featureMapList.append(FeatureMap(self.convOutputSize,poolingType,activationType,batchSize))

    def ClearKernelGrads(self):
        self.KernelGrads = np.zeros((self.numPrevLayerFeatureMaps,self.numFeatureMaps,self.kernelSize,self.kernelSize))
        self.KernelGradsSiamese = np.zeros((self.numPrevLayerFeatureMaps,self.numFeatureMaps,self.kernelSize,self.kernelSize))
        for i in range(0,len(self.featureMapList)):
                fmp = self.featureMapList[i]
                fmp.BiasGradient = 0
                fmp.BiasGradientSiamese = 0

    def Evaluate(self,PrevLayerOutputList,batchIndex):
        # inputs are from the previous layer (unless first layer)
        # convolve inputs with Kernels
        for i in range(0,self.numPrevLayerFeatureMaps):
            for j in range(0,self.numFeatureMaps):
                currentKernel = self.Kernels[i,j]
                self.ConvolResults[batchIndex,i,j] = convolve2d(PrevLayerOutputList[i],currentKernel,mode='valid',boundary='symm')
        # Add Convolution Results
        for q in range(0,len(self.featureMapList)):
            self.ConvolSums[batchIndex,q] = np.zeros((self.convOutputSize[0],self.convOutputSize[1]))
            for p in range(0,len(PrevLayerOutputList)):
                #Sum of all of the convolutions for a given input
                self.ConvolSums[batchIndex,q] += self.ConvolResults[batchIndex,p,q]
        # Evaluate each feature map
        for i in range(0, len(self.featureMapList)):
            self.featureMapList[i].Evaluate(self.ConvolSums[batchIndex,i],batchIndex)

    
    def InitMatrix2DArray(self, Mat, dim1, dim2, matrixSize):
        Mat = np.empty((dim1,dim2))
        for i in range(0,dim1):
            for j in range (0,dim2):
                Mat[i,j] = np.zeros((matrixSize,matrixSize))
    
    def InitializeKernels(self,initRange):
        for i in range (0,self.Kernels.shape[0]):
            for j in range (0,self.Kernels.shape[1]):
                self.Kernels[i,j] = np.random.uniform(low=-initRange,high=initRange,size=(self.kernelSize,self.kernelSize))

