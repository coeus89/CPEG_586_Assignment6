import math
import numpy as np
from numpy.core.defchararray import array
from Layer import *
from MyEnums import *
from CNNLayer import *
from CNNEnums import *
from sklearn.utils import shuffle
from scipy.signal import convolve2d

class Network(object):
    def __init__(self,X,Y,numCNNLayers,kernelSize,poolingType,numLayers,dropout = 1.0, activationFunction = ActivationType.SIGMOID, lastLayerAF = ActivationType.SOFTMAX,batchSize = 1):
        self.X = X
        self.Y = Y
        self.numCNNLayers = numCNNLayers # array in the form [4, 6] for 2 layers of 4 and 6 feature maps respectively
        self.numLayers = numLayers # array in the form [50, 10] for 2 layers of 50 and 10 neurons respectively
        self.dropout = 1.0 - dropout # 0.0 to 1.0 for 0 to 100 percent dropout. (0 to 100 percent zeros for the self.dropout variable)
        self.activationFunction = activationFunction
        self.lastLayerAF = lastLayerAF
        self.Layers = []
        self.myCNNLayers = []
        self.numOfLayers = len(numLayers)
        self.numOfCNNLayers = len(numCNNLayers)
        self.batchSize = batchSize
        self.NNInputSize = 0
        self.Flatten = np.zeros((batchSize), dtype=object)

        # Initialize the CNN Layers
        for j in range(0,len(numCNNLayers)):
            if (j == 0):
                inputSize = self.X.shape[1]
                self.myCNNLayers.append(CNNLayer(self.numCNNLayers[j],1,inputSize,kernelSize,poolingType,activationFunction,batchSize))
            else:
                inputSize = self.X.shape[1]
                for k in range(1, j+1):
                    inputSize = (int)((inputSize - kernelSize + 1)/2) # Make sure the 2nd layer input is 12
                self.myCNNLayers.append(CNNLayer(self.numCNNLayers[j],self.numCNNLayers[j-1],inputSize,kernelSize,poolingType,activationFunction,batchSize))

        # Initialize the Normal Layers
        for i in range(self.numOfLayers):
            if (i == 0):
                # First NN layer coming from a CNN
                prevFeatureMapSize = (self.myCNNLayers[len(self.myCNNLayers) - 1].poolOutputSize)**2
                flattenSize = (prevFeatureMapSize) * self.myCNNLayers[len(self.myCNNLayers) - 1].numFeatureMaps
                # make the layer
                layer = Layer(self.numLayers[i],flattenSize,False,self.dropout,self.activationFunction)
            elif (i == (self.numOfLayers - 1)):
                layer = Layer(self.Y.shape[1],self.numLayers[i-1],True,self.dropout,self.lastLayerAF)
            else:
                layer = Layer(self.numLayers[i],self.numLayers[i-1], False,dropout,self.activationFunction)
            self.Layers.append(layer)
        
        # Find the flattened expected size for the normal NN input
        self.NNInputSize = self.X.shape[1]
        for k in range(1, len(numCNNLayers)+1):
            self.NNInputSize = (self.NNInputSize - kernelSize + 1)/2 # Make sure the 3nd iteration is 4
        self.NNInputSize = (self.NNInputSize**2) * numCNNLayers[len(numCNNLayers)-1] #for 2 layers it should be 4 * 4 * 6 if six feature maps
    
    def Evaluate(self,batch,batchSize,doBatchNorm=False,batchType=BatchNormMode.TEST):
        # Evaluate CNN Layers
        # Note: batch is in format [batchSize,prevFeatureMaps,FeatureMapWidth,FeatureMapHeight]
        # prevFeatureMaps may just be the single image. so will be 1 on first layer.
        for j in range(0, len(self.numCNNLayers)): # select Layer
            PrevOut = None
            if (j == 0):
                PrevOut = batch 
            else:
                poolOutputSize = self.myCNNLayers[j - 1].poolOutputSize
                PrevOut = np.zeros((batchSize, self.myCNNLayers[j-1].numFeatureMaps,poolOutputSize,poolOutputSize))
                for k in range(0,len(self.myCNNLayers[j-1].featureMapList)): # select Feature Map
                    BatchFeatureMapOut = self.myCNNLayers[j - 1].featureMapList[k].OutputPool
                    #for m in range(0, len(BatchFeatureMapOut)): # This puts the prevOut in the format [batch,featureMapOutput] for a batch of 5 and 4 feature maps it will be 5x4x12x12
                    for m in range(0, batchSize): # This puts the prevOut in the format [batch,featureMapOutput] for a batch of 5 and 4 feature maps it will be 5x4x12x12
                        PrevOut[m,k] = BatchFeatureMapOut[m]
                
            # For each item in batch evaluate
            for batchIndex in range(0,batchSize):
                PreviousOutput = PrevOut[batchIndex]
                self.myCNNLayers[j].Evaluate(PreviousOutput,batchIndex)
        
        
        # Flatten the last CNN output
        # get the Feature Map Output Pools into the format [batch,OutputVectors] which should be a 5 x 6 for a batch size of 5 and a feature map size of 6.
        featureMapSize = (self.myCNNLayers[len(self.myCNNLayers) - 1].poolOutputSize)**2
        flattenSize = (featureMapSize) * self.myCNNLayers[len(self.myCNNLayers) - 1].numFeatureMaps
        self.Flatten = np.zeros((batchSize,flattenSize))
        for bIndex in range(0,batchSize):
            flatFM = []
            for fmIndex in range(0,self.myCNNLayers[len(self.myCNNLayers) - 1].numFeatureMaps):
                fm = self.myCNNLayers[len(self.myCNNLayers) - 1].featureMapList[fmIndex].OutputPool[bIndex]
                flatFM = np.append(flatFM,fm.reshape(featureMapSize))
                test = ""
            self.Flatten[bIndex] = flatFM


        # Normal NN Layers' Execution
        currBatch = self.Flatten
        self.Layers[0].Evaluate(currBatch,doBatchNorm,batchType)
        for i in range(1,self.numOfLayers):
            self.Layers[i].Evaluate(self.Layers[i-1].a,doBatchNorm,batchType)
        return self.Layers[self.numOfLayers - 1].a

    def BackProp(self,batch_x,batch_y,layerNumber,batchSize,doBatchNorm,batchType):
        # Normal NN Layers' Back Prop
        layer = self.Layers[layerNumber]
        if (layerNumber == (self.numOfLayers - 1)): #last layer
            if (layer.activationType == ActivationType.SOFTMAX):
                layer.SoftMaxDeltaLL(batch_y, batchSize, doBatchNorm, batchType) #last layer
            else:
                layer.CalcDeltaLL(batch_y, batchSize, doBatchNorm, batchType) #last layer
        else:
            layer.CalcDelta(self.Layers[layerNumber + 1].deltabn, self.Layers[layerNumber + 1].w, batchSize, doBatchNorm, batchType)
        
    def CNNBackProp(self,layerNumber,X_Data):
        # CNN Layers' Back Prop
        # Remember to do for each element of the batch.
        # Add gradients together later.
        deltaFlatten = None
        currCNNLayer = self.myCNNLayers[layerNumber]
        currNumFeatureMaps = currCNNLayer.numFeatureMaps

        if (layerNumber == (self.numOfCNNLayers - 1)): # Last CNN Layer requires unflattening first.
            deltaFlatten = self.Layers[0].CalcDeltaFlatten()
            #currCNNLayer = self.myCNNLayers[self.numOfCNNLayers - 1]
            poolOutputSize = currCNNLayer.poolOutputSize
            pool2 = poolOutputSize**2
            #res = np.zeros((self.batchSize,currNumFeatureMaps,poolOutputSize,poolOutputSize))
            for i in range(0,self.batchSize):
                for j in range(0,currNumFeatureMaps):
                    #res[i,j] = deltaFlatten[i,j*pool2:j*pool2 + pool2].reshape(poolOutputSize,poolOutputSize)
                    #Calc DeltaPool
                    currCNNLayer.featureMapList[j].DeltaPool[i] = deltaFlatten[i,j*pool2:j*pool2 + pool2].reshape(poolOutputSize,poolOutputSize)
        else:
            temp = "Pause" #this needs to become the general case. Calc? the delta pool from previous layer.
            #Get DeltaPool via convolution
        #Calc DeltaCV
        for i in range(0,self.batchSize):
            for j in range(0,currNumFeatureMaps):
                fmp = currCNNLayer.featureMapList[j]
                fmp.DeltaCV[i] = np.zeros((fmp.OutputPool[i].shape[0] * 2, fmp.OutputPool[i].shape[1] * 2))
                indexM = 0
                indexN = 0
                for m in range(0,fmp.DeltaPool[i].shape[0]):
                    indexN = 0
                    for n in range(0,fmp.DeltaPool[i].shape[1]):
                        if (fmp.activationType == ActivationType.SIGMOID or fmp.activationType == ActivationType.TANH):
                            fmp.DeltaCV[i,indexM,indexN] = (1/4.) * fmp.DeltaPool[i,m,n] * fmp.APrime[i,indexM,indexN]
                            fmp.DeltaCV[i,indexM,indexN + 1] = (1/4.) * fmp.DeltaPool[i,m,n] * fmp.APrime[i,indexM,indexN + 1]
                            fmp.DeltaCV[i,indexM + 1,indexN] = (1/4.) * fmp.DeltaPool[i,m,n] * fmp.APrime[i,indexM + 1,indexN]
                            fmp.DeltaCV[i,indexM + 1,indexN + 1] = (1/4.) * fmp.DeltaPool[i,m,n] * fmp.APrime[i,indexM + 1,indexN + 1]
                            indexN += 2
                        if (fmp.activationType == ActivationType.RELU):
                            if (fmp.Sum[i,indexM,indexN] > 0):
                                fmp.DeltaCV[i,indexM,indexN] = (1/4.) * fmp.DeltaPool[i,m,n]
                            else:
                                fmp.DeltaCV[i,indexM,indexN] = 0
                            if (fmp.Sum[i,indexM,indexN + 1] > 0):
                                fmp.DeltaCV[i,indexM,indexN + 1] = (1/4.) * fmp.DeltaPool[i,m,n]
                            else:
                                fmp.DeltaCV[i,indexM,indexN + 1] = 0
                            if (fmp.Sum[i,indexM + 1,indexN] > 0):
                                fmp.DeltaCV[i,indexM + 1,indexN] = (1/4.) * fmp.DeltaPool[i,m,n]
                            else:
                                fmp.DeltaCV[i,indexM + 1,indexN] = 0
                            if (fmp.Sum[i,indexM + 1,indexN + 1] > 0):
                                fmp.DeltaCV[i,indexM + 1,indexN + 1] = (1/4.) * fmp.DeltaPool[i,m,n]
                            else:
                                fmp.DeltaCV[i,indexM + 1,indexN + 1] = 0
                    indexM += 2
            #Calc Gradients for bias.
            for f in range(0,currNumFeatureMaps):
                fmp = currCNNLayer.featureMapList[f]
                for u in range(0, fmp.DeltaCV[i].shape[0]):
                    for v in range(0, fmp.DeltaCV[i].shape[1]):
                        fmp.BiasGradient += fmp.DeltaCV[i,u,v]
            
            #Calc Gradients for pxq kernels in the current layer
            numFeaturesThisLayer = self.numCNNLayers[layerNumber]
            numFeaturesPrevLayer = self.numCNNLayers[layerNumber-1]
            if (layerNumber > 0): #not the first layer
                # Find Gradients for Kernels
                for p in range(0,numFeaturesPrevLayer):
                    for q in range(0,numFeaturesThisLayer):
                        #check this. Maybe break it up
                        part1 = np.rot90(self.myCNNLayers[layerNumber - 1].featureMapList[p].OutputPool[i],2)
                        part2 = self.myCNNLayers[layerNumber].featureMapList[q].DeltaCV[i]
                        self.myCNNLayers[layerNumber].KernelGrads[p,q] = self.myCNNLayers[layerNumber].KernelGrads[p,q] + convolve2d(part1,part2,mode='valid',boundary='symm')       
                # Find deltaPool for previous Layer
                for p in range(0,numFeaturesPrevLayer):
                    size = self.myCNNLayers[layerNumber - 1].featureMapList[p].OutputPool[i].shape[0]
                    self.myCNNLayers[layerNumber - 1].featureMapList[p].DeltaPool[i] = np.zeros((size,size))
                    for q in range(0,numFeaturesThisLayer):
                        # Prev Layer DeltaPool
                        part1 = self.myCNNLayers[layerNumber].featureMapList[p].DeltaCV[i]
                        part2 = np.rot90(self.myCNNLayers[layerNumber].Kernels[p,q],2)
                        self.myCNNLayers[layerNumber - 1].featureMapList[p].DeltaPool[i] += convolve2d(part1,part2,mode='full',boundary='symm')
            else:
                # This is first layer attached to input
                for p in range(0,1):
                    for q in range(0,numFeaturesThisLayer):
                        # check this. Maybe break it up
                        # The first layer is just the input. Just figuring out the kernelgrads
                        part1 = np.rot90(X_Data[i,0],2)
                        part2 = self.myCNNLayers[layerNumber].featureMapList[q].DeltaCV[i]
                        self.myCNNLayers[layerNumber].KernelGrads[p,q] += convolve2d(part1,part2,mode='valid',boundary='symm') 
            
    def Train(self,Epochs,LearningRate,doBatchNorm = False,lroptimization = LROptimizerType.NONE):
        for ep in range(0,Epochs):
            loss = 0
            itnum = 0
            self.X, self.Y = shuffle(self.X, self.Y, random_state=0)
            # shuffle(self.X, self.Y, random_state=0)

            # Evaluate CNN and Normal Layers
            for batch_i in range(0, self.X.shape[0], self.batchSize):
                batch_x = self.X[batch_i:batch_i + self.batchSize]
                batch_y = self.Y[batch_i:batch_i + self.batchSize]
                # Need to add the 1 into the array so that the CNN likes the shape
                batch_x = batch_x.reshape(batch_x.shape[0],1,batch_x.shape[1],batch_x.shape[2])
                #batch_y = batch_y.reshape(batch_y.shape[0],1,batch_y.shape[1],batch_y.shape[2])
                LLa = self.Evaluate(batch_x,self.batchSize,doBatchNorm,BatchNormMode.TRAIN) # Last Layer 'a' value
                
                if (self.lastLayerAF == ActivationType.SOFTMAX):
                    # Use cross entropy loss
                    loss += (-batch_y*np.log(LLa)).sum()
                else:
                    # use mean square loss
                    loss += (0.5 * (batch_y - LLa)**2)

                # Calc NN Deltas
                layerNumber = self.numOfLayers - 1
                while (layerNumber >= 0):
                    self.BackProp(batch_x,batch_y,layerNumber,self.batchSize,doBatchNorm,BatchNormMode.TRAIN)
                    self.CalcGradients(layerNumber,batch_x)
                    layerNumber -= 1

                # Calc CNN Deltas
                CNNLayerNumber = self.numOfCNNLayers - 1
                while (CNNLayerNumber >= 0):
                    self.CNNBackProp(CNNLayerNumber,batch_x)
                    CNNLayerNumber -= 1

                itnum += 1
                self.UpdateGradsBiases(LearningRate,self.batchSize,lroptimization,itnum,doBatchNorm)
                self.UpdateCNNKernelsBiases(LearningRate,self.batchSize)
                self.clearCNNGradients()
            print("Epoch: " + str(ep) + ",   Loss: "+ str(loss))
    
    def CalcGradients(self,layerNumber,batch_x):
        # Calculate NN Gradients
        if (layerNumber > 0):
            prevOut = self.Layers[layerNumber - 1].a
        else:
            # prevOut = batch_x
            prevOut = self.Flatten
        self.Layers[layerNumber].CalcGradients(prevOut)


    def UpdateGradsBiases(self, learningRate, batchSize, LROptimization, itnum, doBatchNorm):
        # update weights and biases for all layers
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        for ln in range(len(self.numLayers)):
            if (LROptimization == LROptimizerType.NONE):
                self.Layers[ln].UpdateWb(learningRate, batchSize,doBatchNorm)
            elif (LROptimization == LROptimizerType.ADAM):
                self.Layers[ln].CalcAdam(itnum,learningRate,batchSize,doBatchNorm,beta1,beta2,epsilon)
    
    def UpdateCNNKernelsBiases(self, learningRate, batchSize):
        for cnnCount in range(0,len(self.numCNNLayers)):
            if (cnnCount == 0):
                for p in range(0,1):
                    for q in range(0,len(self.myCNNLayers[0].featureMapList)):
                        self.myCNNLayers[cnnCount].Kernels[p,q] -= self.myCNNLayers[cnnCount].KernelGrads[p,q] * (1./batchSize) * learningRate
            else:
                for p in range (0,len(self.myCNNLayers[cnnCount - 1].featureMapList)):
                    for q in range(0,len(self.myCNNLayers[cnnCount].featureMapList)):
                        self.myCNNLayers[cnnCount].Kernels[p,q] -= self.myCNNLayers[cnnCount].KernelGrads[p,q] * (1./batchSize) * learningRate
            for i in range(0,len(self.myCNNLayers[cnnCount].featureMapList)):
                fmp = self.myCNNLayers[cnnCount].featureMapList[i]
                fmp.Bias -= fmp.BiasGradient * (1./batchSize) * learningRate
    
    def clearCNNGradients(self):
        for cnnCount in range(0,len(self.numCNNLayers)):
            self.myCNNLayers[cnnCount].ClearKernelGrads()