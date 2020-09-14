import tensorflow as tf
import numpy as np
import sys
from collections import deque
import os
import gym
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from env.generateReberStr import *
from src.buildLSTM import *

groundTruth = [[1, 2], [0, 1, 3, 4, 5, 6], [0, 1, 3, 5, 6], [3, 4, 6], [1, 3, 4, 5], [2, 5, 6], [0, 1, 2, 3, 4, 5, 6]]

stateDim = 7
actionDim = 7
learningRate = 0.001
numStrings = 256
layerWidths = [10]
trainingRounds = 1000

buildModel = BuildModel(stateDim, actionDim)
rebersTrain = embeddedReberGrammar(reberGrammar(numStrings))
model = buildModel(layerWidths, layerWidths)
xData, yData = translateReberToNumber(rebersTrain)
statesBatch, yBatch = transformToOneHot(xData, yData, stateDim)
batchSize = len(statesBatch)
trainOneStep = TrainOneStep(learningRate, actionDim, batchSize)

runAlgorithm = RunAlgorithm(learningRate, actionDim, batchSize, trainingRounds, trainOneStep)
model, loss, cell, output, LOSS, ACC = runAlgorithm(model, statesBatch, yBatch)

rebersTest = embeddedReberGrammar(reberGrammar(numStrings*2))
xData, yData = translateReberToNumber(rebersTest)
statesBatch, yBatch = transformToOneHot(xData, yData, stateDim)
statesBatch = statesBatch[0: batchSize]
yBatch = yBatch[0: batchSize]
accuracy = runTestSet(model, statesBatch, output, cell)
print(accuracy)

import matplotlib.pyplot as plt

plt.plot(LOSS, color='green')
plt.show()
plt.plot(ACC, color='red')
plt.show()