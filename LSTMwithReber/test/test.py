import tensorflow as tf
import numpy as np
import sys
from ddt import ddt, data, unpack
from collections import deque
import os
import gym
import random
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))
import unittest

from src.buildModelLSTM import *

@ddt
class TestDQN(unittest.TestCase):
    def setUp(self):

        self.actionDim = 2
        self.stateDim = 4
        self.gamma = 0.9
        self.batchSize = 2
        self.buildModel = BuildModel(self.stateDim, self.actionDim, self.gamma)
        self.layerWidths = [30, 30]
        self.learningRate = 0.001
        self.updateFrequency = 10

    @data(
        ([[[[1,1,0,1], [1,2,3,1]], [[0, 1],[0,1]], [[1.02, 1.19512195, 0.02, 0.70731707],[4,2,0,1]], [[1],[1]], [[0],[0]]]])
    )
    @unpack
    def testImprovement(self, miniBatch):
        buildModel = BuildModel(self.stateDim, self.actionDim, self.gamma)
        model = buildModel(self.layerWidths, self.layerWidths)
        calculateY = CalculateY(model, self.updateFrequency)
        trainOneStep = TrainOneStep(self.batchSize, self.updateFrequency, self.learningRate, self.gamma, calculateY, self.actionDim)
        model, lossWithTrain1 = trainOneStep(model, miniBatch, self.batchSize)
        model, lossWithTrain2 = trainOneStep(model, miniBatch, self.batchSize)
        self.assertTrue(lossWithTrain1 > lossWithTrain2)


if __name__ == '__main__':
    unittest.main()