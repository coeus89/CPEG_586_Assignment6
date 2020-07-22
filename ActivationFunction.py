from os import stat
import numpy as np

class ActivationFunction(object):
    Epsillon = 1.0E-6

    @staticmethod
    def Sigmoid(s):
        # a = np.zeros((s.shape[0],s.shape[1]))
        # derivAF = np.zeros((s.shape[0],s.shape[1]))
        # for x in range(0,s.shape[0]):
        #     for y in range(0,s.shape[1]):
        #         a[x,y] = 1.0/(1.0 + np.exp(-s[x,y]))
        #         derivAF[x,y] = a[x,y] * (1. - a[x,y])
        a = 1.0/(1.0 + np.exp(-s))
        derivAF = a * (1. - a)
        return a, derivAF

    @staticmethod
    def TanH(s):
        a = np.tanh(s)
        derivAF = (1. - a ** 2)
        return a, derivAF

    @staticmethod
    def Relu(s):
        a = np.maximum(0,s)
        #self.derivAF = 1.0 * (self.a > 0)
        derivAF = 1. * (a > ActivationFunction.Epsillon)
        derivAF[derivAF == 0] = ActivationFunction.Epsillon
        return a, derivAF

    # Go back and look at this.
    @staticmethod
    def Softmax(x):
        ex = np.exp(x)
        a = None
        derivAF = None # softmax derivative is done at network layer
        if (x.shape[0] == x.size):
            a = ex/ex.sum()
        for i in range(ex.shape[0]):
            denom = ex[i,:].sum()
            ex[i,:] = ex[i,:]/denom
            a = ex
        return a, derivAF