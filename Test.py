import numpy as np
from Pooling import *
from CNNLayer import *
from CNNEnums import *

# Test1 = np.random.uniform(low=0,high=10,size=(10,10))
# print(str(Test1))
# res = Pooling.AvgPool(Test1)
# print(str(res))


# TestIn = np.random.randint(low=0,high=255,size=(28,28))
# TestArray = []
# TestArray.append(TestIn)
# Test2 = CNNLayer(4,1,28,5,PoolingType.AVGPOOLING,ActivationType.SIGMOID,batchSize=5)
# Test2.Evaluate(TestArray,0)

# numlayers = 2
# kernelSize = 5
# inputSize = 28
# print(str(inputSize))
# for k in range(1, numlayers):
#     inputSize = (int)((inputSize - kernelSize + 1)/2)
#     print(str(inputSize))

Test3 = np.random.uniform(low=0,high=10,size=(4,4))
print(Test3)
Test4 = Test3.reshape((16))
print(Test4)
Test5 = Test4.reshape(Test3.shape)
print(Test5)


pause = ""