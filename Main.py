import os
import sys
import cv2
import math
import numpy as np
from Network import *
from MyEnums import *


def main():
    temp = np.array(["this is a string","This is another staring"])
    tempFileName = (os.listdir('C:\\ATTFaceDataSet\\Training'))[0]
    tempFile = cv2.imread('C:\\ATTFaceDataSet\\Training\\{0}'.format(tempFileName))
    width = tempFile.shape[1]
    height = tempFile.shape[0]
    trainingImages = len(os.listdir("C:\\ATTFaceDataSet\\Training"))
    testImages = len(os.listdir("C:\\ATTFaceDataSet\\Testing"))
    train = np.empty((trainingImages,height,width),dtype=np.float)
    trainY = np.empty((trainingImages),dtype='<U23') # This means string in numpy apparently
    test = np.empty((testImages,height,width),dtype=np.float)
    testY = np.empty((trainingImages),dtype='<U23') # This means string in numpy apparently
    
    #load images
    i = 0
    for filename in os.listdir("C:\\ATTFaceDataSet\\Training"):
        y = filename.split('_')[0]
        trainY[i] = str(y)
        train[i] = cv2.imread('C:\\ATTFaceDataSet\\Training\\{0}'.format(filename),0)/255.0 #for color use 1
        i += 1

    j = 0
    for filename in os.listdir("C:\\ATTFaceDataSet\\Testing"):
        y = filename.split('_')[0]
        testY[j] = str(y)
        test[j] = cv2.imread('C:\\ATTFaceDataSet\\Testing\\{0}'.format(filename),0)/255.0 
        j += 1

    trainX = train#.reshape(train.shape[0],train.shape[1]*train.shape[2])
    testX = test#.reshape(test.shape[0],test.shape[1]*test.shape[2])


    # Configuration Start for Siamese Network

    numCNNLayers = [6,12] # Number of deep cnn layers
    numLayers = [100] # Number of classification layers = 1 & Number of Neurons = 100
    # Should NN Layer be Softmax???

    dropOut = 1.0 #20% dropout
    hiddinActivation = ActivationType.RELU
    LLActivation = ActivationType.SOFTMAX
    kernelSize = 5
    poolingType = PoolingType.AVGPOOLING
    batchSize = 1 # This is supposed to be stochastic

    myNetwork = Network(trainX,trainY,numCNNLayers,kernelSize,poolingType,numLayers,dropOut,hiddinActivation,LLActivation,batchSize)

    epochs = 30
    learningRate = 0.1
    #lambda1 = 0. #don't use. not sure why it's there.
    #trainType = GradDescType.MiniBatch
    
    doBatchNorm = False
    lropt = LROptimizerType.NONE
    # #-------------------For Testing, delete later----------------
    # myNetwork.Evaluate(trainX[0:5].reshape(5,1,28,28))
    # #------------------------------------------------------------
    myNetwork.Train(epochs,learningRate,doBatchNorm,lropt)

    print("Finished Training. \nTesting Begins")

    accuracyCount = 0
    testX = testX.reshape(testX.shape[0],1,1,testX.shape[1],testX.shape[2])

    for i in range(testY.shape[0]):
        a2 = myNetwork.Evaluate(testX[i],1,doBatchNorm,BatchNormMode.TEST)
        maxindex = a2[0].argmax(axis = 0)
        if (testY[i,maxindex] == 1):
            accuracyCount = accuracyCount + 1
    print("Accuracy count = " + str(accuracyCount/testY.shape[0]*100) + '%')

if __name__ == "__main__":
    sys.exit(int(main() or 0))