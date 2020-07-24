import numpy as np
import math
from MyEnums import ActivationType, BatchNormMode
from sklearn.utils import shuffle
#from MyEnums import BatchNormMode

class Layer(object):
    Epsillon = 1.0E-6
    def __init__(self, numNeurons,numNeuronsPrevLayer,lastLayer=False,dropout = 0.2, activationType = ActivationType.SIGMOID):
        self.numNeurons = numNeurons
        self.numNeuronsPrevLayer = numNeuronsPrevLayer
        self.lastLayer = lastLayer
        self.dropout = dropout
        self.activationType = activationType
        self.learningRate = 0.1
        self.w = np.random.uniform(low=-0.1,high=0.1,size=(numNeurons,numNeuronsPrevLayer))
        self.b = np.random.uniform(low=-1.0,high=1.0,size=(numNeurons))
        self.gradw = np.zeros((numNeurons,numNeuronsPrevLayer))
        self.gradb = np.zeros((numNeurons))
        self.delta = np.zeros((numNeurons))
        self.a = np.zeros((numNeurons))
        self.derivAF = np.zeros((numNeurons)) # deriv of Activation function
        self.zeroout = None #for dropout
        # For Siamese
        self.gradwSiamese = np.zeros((numNeurons,numNeuronsPrevLayer))
        self.gradbSiamese = np.zeros((numNeurons))
        # For Adam
        self.mtw = np.zeros((self.w.shape))
        self.mtb = np.zeros((self.b.shape))
        self.vtw = np.zeros((self.w.shape))
        self.vtb = np.zeros((self.b.shape))
        # For BatchNorm
        self.batchCounter = 0
        self.runningMean = np.zeros((numNeurons))
        self.runningVariance = np.zeros((numNeurons))
        self.BatchMean = np.zeros((numNeurons))
        self.BatchVariance = np.zeros((numNeurons))
        self.gamma = np.random.rand(1) #np.random.uniform(low=-0.1,high=0.1,size=(numNeurons))
        self.dgamma = np.zeros((numNeurons,1))
        self.beta = np.random.rand(1) #np.random.uniform(low=-1.0,high=1.0,size=(numNeurons))
        self.dbeta = np.zeros((numNeurons))
        self.deltabn = np.zeros((numNeurons))
        self.sihat = np.zeros((numNeurons))
        self.si = np.zeros((numNeurons))
        self.sb = np.zeros((numNeurons))
        # For Future CNN Back Prop
        self.deltaFlatten = np.zeros((numNeuronsPrevLayer))
        
    def Evaluate(self,indata,doBatchNorm,bNormMode):
        self.si = np.dot(indata,self.w.T) + self.b
        if (doBatchNorm == True):
            if (bNormMode == BatchNormMode.TRAIN):
                self.GetBatchMeanVar(self.si)
            else:
                self.BatchMean = self.runningMean
                self.BatchVariance = self.runningVariance

            self.sihat = (self.si - self.BatchMean) / (np.sqrt(self.BatchVariance + Layer.Epsillon))
            self.sb = self.sihat * self.gamma + self.beta
        else:
            self.sb = self.si

        if (self.activationType == ActivationType.SIGMOID):
            self.a = self.Sigmoid(self.sb)
            self.derivAF = self.a * (1 - self.a)
        if (self.activationType == ActivationType.TANH):
            self.a = self.TanH(self.sb)
            self.derivAF = (1 - self.a ** 2)
        if (self.activationType == ActivationType.RELU):
            self.a = self.Relu(self.sb)
            #self.derivAF = 1.0 * (self.a > 0)
            self.derivAF = 1. * (self.a > Layer.Epsillon)
            self.derivAF[self.derivAF == 0] = Layer.Epsillon
        if (self.activationType == ActivationType.SOFTMAX):
            self.a = self.Softmax(self.sb)
            self.derivAF = None # we do delta computation in Network layer
            stop = "stop here"
        if (self.lastLayer == False):
            # get array ready for X% dropout. 80% dropout = 20% zeros
            dropoutZeros = math.ceil(self.dropout*self.numNeurons)
            dropoutOnes = self.numNeurons - dropoutZeros
            ZerosVector = np.zeros((dropoutZeros))
            OnesVector = np.ones((dropoutOnes))
            self.zeroout = shuffle(np.concatenate((ZerosVector,OnesVector)))
            #self.zeroout = np.random.binomial(1,self.dropout,(self.numNeurons,1))/self.dropout
            self.a = self.a * self.zeroout #do the dropout
            self.derivAF = self.derivAF * self.zeroout #do the dropout

    def Sigmoid(self,s):
        return 1.0/(1.0 + np.exp(-s))

    def TanH(self,s):
        return np.tanh(s)

    def Relu(self,s):
        return np.maximum(0,s)

    # Go back and look at this.
    def Softmax(self, x):
        if (x.shape[0] == x.size):
            ex = np.exp(x)
            return ex/ex.sum()
        ex = np.exp(x)
        for i in range(ex.shape[0]):
            denom = ex[i,:].sum()
            ex[i,:] = ex[i,:]/denom
        return ex
    
    
    def CalcAdam(self, itnum, learningRate, batchSize, doBatchNorm, Beta1 = 0.9, Beta2 = 0.999, epsilon = 1e-6,UpdateSiameseWB = False):
        WGrad = self.gradw
        BGrad = self.gradb
        if(UpdateSiameseWB == True):
            WGrad = self.gradw - self.gradwSiamese
            BGrad = self.gradb - self.gradbSiamese
        self.mtw = Beta1 * self.mtw + (1 - Beta1)*WGrad
        self.mtb = Beta1 * self.mtb + (1 - Beta1)*BGrad
        self.vtw = Beta2 * self.vtw + (1 - Beta2)*WGrad*WGrad
        self.vtb = Beta2 * self.vtb + (1 - Beta2)*BGrad*BGrad

        mtwhat = self.mtw / (1 - Beta1**itnum)
        mtbhat = self.mtb / (1 - Beta1**itnum)
        vtwhat = self.vtw / (1 - Beta2**itnum)
        vtbhat = self.vtb / (1 - Beta2**itnum)

        self.w = self.w - learningRate * (1./batchSize) * mtwhat /((vtwhat**0.5) + epsilon)
        self.b = self.b - learningRate * (1./batchSize) * mtbhat /((vtbhat**0.5) + epsilon)

        if (doBatchNorm == True):
            self.UpdateBetaGamma(learningRate)
    
    def UpdateWb(self, learningRate, batchSize, doBatchNorm):
        self.w = self.w - learningRate * (1/batchSize) * (self.gradw - self.gradwSiamese)#- learningRate * lambda1 * self.w.sum()
        self.b = self.b - learningRate * (1/batchSize) * (self.gradb - self.gradbSiamese)
        if (doBatchNorm == True):
            self.UpdateBetaGamma(learningRate)
    
    def GetBatchMeanVar(self, indata):
        self.BatchMean = np.mean(indata,axis=0)
        self.BatchVariance = np.var(indata,axis=0)
        self.runningMean = 0.9 * self.runningMean + (1.0 - 0.9) * self.BatchMean
        self.runningVariance = 0.9 * self.runningVariance + (1.0 - 0.9) * self.BatchVariance

    def SoftMaxDeltaLL(self, batch_y, batchSize, doBatchNorm, batchType): #Last Layer
        self.delta = self.a - batch_y
        if (doBatchNorm == True):
            self.CalcBatchBackProp(batchSize,doBatchNorm,batchType)
        else:
            self.deltabn = self.delta
    
    def CalcDeltaLL(self, batch_y, batchSize, doBatchNorm, batchType): #Last Layer
        self.delta = -(batch_y - self.a) * self.derivAF
        if (doBatchNorm == True):
            self.CalcBatchNormBackProp(self.delta,batchSize,doBatchNorm,batchType)
        else:
            self.deltabn = self.delta

    def CalcDelta(self, delta_NextLayer, W_NextLayer, batchSize, doBatchNorm, batchType): #other layers
        self.delta = np.dot(delta_NextLayer,W_NextLayer) * self.derivAF
        if (doBatchNorm == True):
            self.CalcBatchBackProp(batchSize,doBatchNorm,batchType)
        else:
            self.deltabn = self.delta
            
    def CalcBatchBackProp(self, batchSize, doBatchNorm, batchType):
        self.dgamma = np.sum(self.delta * self.sihat,axis=0)
        self.dbeta = np.sum(self.delta,axis=0)
        self.deltabn = (self.delta * self.gamma) / (batchSize * np.sqrt(self.BatchVariance + Layer.Epsillon)) * (batchSize - 1 - (self.sihat * self.sihat))

    def CalcGradients(self, prevOut, UpdateSiameseWB = False):
        if (UpdateSiameseWB == False):
            self.gradw = np.dot(self.deltabn.T,prevOut)
            self.gradb = self.deltabn.sum(axis=0)
        else:
            self.gradwSiamese = np.dot(self.deltabn.T,prevOut)
            self.gradbSiamese = self.deltabn.sum(axis=0)
        #it shouldnt matter if it's batch norm or not because we set deltabn = delta in CalcDelta method
    
    def UpdateBetaGamma(self, learningRate): #only for batch norm
        self.beta = self.beta - learningRate * self.dbeta
        self.gamma = self.gamma - learningRate * self.dgamma
    
    def CalcDeltaFlatten(self):
        self.deltaFlatten = np.dot(self.deltabn,self.w)
        return self.deltaFlatten
