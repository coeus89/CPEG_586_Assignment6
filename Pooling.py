import numpy as np

class Pooling(object):
    @staticmethod
    def AvgPool(inputMatrix):
        res = np.zeros(((int)(inputMatrix.shape[0]/2),(int)(inputMatrix.shape[1]/2)))
        for i in range(0,res.shape[0]):
            for j in range(0,res.shape[1]):
                res[i,j] = (inputMatrix[2 * i,2 * j] + inputMatrix[2 * i + 1,2 * j] + inputMatrix[2 * i,2 * j + 1] + inputMatrix[2 * i + 1,2 * j + 1])/4.0
        return res
    
    @staticmethod
    def MaxPool(inputMatrix):
        res = np.zeros(((int)(inputMatrix.shape[0]/2),(int)(inputMatrix.shape[1]/2)))
        for i in range(0,res.shape[0]):
            for j in range(0,res.shape[1]):
                res[i,j] = np.max(np.array([inputMatrix[2 * i,2 * j] , inputMatrix[2 * i + 1,2 * j] , inputMatrix[2 * i,2 * j + 1] , inputMatrix[2 * i + 1,2 * j + 1]]))
        return res
    
    @staticmethod
    def MinPool(inputMatrix):
        res = np.zeros(((int)(inputMatrix.shape[0]/2),(int)(inputMatrix.shape[1]/2)))
        for i in range(0,res.shape[0]):
            for j in range(0,res.shape[1]):
                res[i,j] = np.min(np.array([inputMatrix[2 * i,2 * j] , inputMatrix[2 * i + 1,2 * j] , inputMatrix[2 * i,2 * j + 1] , inputMatrix[2 * i + 1,2 * j + 1]]))
        return res
