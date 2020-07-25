            #_______________OLD_______________
            # Evaluate CNN and Normal Layers
            # for batch_i in range(0, self.X.shape[0], self.batchSize):
            #     batch_x = self.X[batch_i:batch_i + self.batchSize]
            #     batch_y = self.Y[batch_i:batch_i + self.batchSize]
            #     # Need to add the 1 into the array so that the CNN likes the shape
            #     batch_x = batch_x.reshape(batch_x.shape[0],1,batch_x.shape[1],batch_x.shape[2])
            #     #batch_y = batch_y.reshape(batch_y.shape[0],1,batch_y.shape[1],batch_y.shape[2])
            #     LLa = self.Evaluate(batch_x,self.batchSize,doBatchNorm,BatchNormMode.TRAIN) # Last Layer 'a' value
                
            #     if (self.lastLayerAF == ActivationType.SOFTMAX):
            #         # Use cross entropy loss
            #         loss += (-batch_y*np.log(LLa)).sum()
            #     else:
            #         # use mean square loss
            #         loss += (0.5 * (batch_y - LLa)**2)

            #     # Calc NN Deltas
            #     layerNumber = self.numOfLayers - 1
            #     while (layerNumber >= 0):
            #         self.BackProp(batch_x,batch_y,layerNumber,self.batchSize,doBatchNorm,BatchNormMode.TRAIN)
            #         self.CalcGradients(layerNumber)
            #         layerNumber -= 1

            #     Calc CNN Deltas
            #     CNNLayerNumber = self.numOfCNNLayers - 1
            #     while (CNNLayerNumber >= 0):
            #         self.CNNBackProp(CNNLayerNumber,batch_x)
            #         CNNLayerNumber -= 1

            #     itnum += 1
            #     self.UpdateGradsBiases(LearningRate,self.batchSize,lroptimization,itnum,doBatchNorm)
            #     self.UpdateCNNKernelsBiases(LearningRate,self.batchSize)
            #     self.clearCNNGradients()
