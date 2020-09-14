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

groundTruth = [[1, 2], [0, 1, 3, 4, 5, 6], [0, 1, 3, 5, 6], [3, 4, 6], [1, 3, 4, 5], [2, 5, 6], [0, 1, 2, 3, 4, 5, 6]]


class BuildModel:
    def __init__(self, numStateSpace, numActionSpace, seed=1):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.seed = seed

    def __call__(self, sigmoidLayersWidths, tanhLayerWidths, summaryPath="./tbdata"):
        print("Generating LSTM Model with layers: {}, {}".format(sigmoidLayersWidths, tanhLayerWidths))
        graph = tf.Graph()
        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            with tf.name_scope('inputs'):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="states")
                formerOutput_ = tf.placeholder(tf.float32, [None, self.numActionSpace], name="formerOutput")
                formerCell_ = tf.placeholder(tf.float32, [None, self.numActionSpace], name="formerCell")
                yi_ = tf.placeholder(tf.float32, [None, self.numActionSpace], name="yi")
                tf.add_to_collection("states", states_)
                tf.add_to_collection("yi", yi_)
                tf.add_to_collection("formerCell", formerCell_)
                tf.add_to_collection("formerOutput", formerOutput_)

            inputStates_ = tf.concat([formerOutput_, states_], 1)
            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.01)

            with tf.variable_scope("forgetSigmoidGate"):
                with tf.variable_scope("trainForgetHidden"):
                    activation_ = inputStates_
                    for i in range(len(sigmoidLayersWidths)):
                        forgetHiddenLayer = tf.layers.Dense(units=sigmoidLayersWidths[i], activation=None,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="forgetHidden{}".format(i + 1),
                                                  trainable=True)
                        activation_ = forgetHiddenLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{forgetHiddenLayer.kernel.name}"], forgetHiddenLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{forgetHiddenLayer.bias.name}"], forgetHiddenLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    forgetHiddenOutput_ = tf.identity(activation_, name="forgetHiddenOutput")
                    forgetOutputLayer = tf.layers.Dense(units=self.numActionSpace, activation=tf.sigmoid,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="forgetOutputLayer{}".format(len(sigmoidLayersWidths) + 1),
                                                        trainable=True)
                    forgetOutput_ = forgetOutputLayer(forgetHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{forgetOutputLayer.kernel.name}"], forgetOutputLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{forgetOutputLayer.bias.name}"], forgetOutputLayer.bias)
                    tf.add_to_collections("forgetOutput", forgetOutput_)

            with tf.variable_scope("inputSigmoidGate"):
                with tf.variable_scope("trainInputHidden"):
                    activation_ = inputStates_
                    for i in range(len(sigmoidLayersWidths)):
                        inputHiddenLayer = tf.layers.Dense(units=sigmoidLayersWidths[i], activation=None,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="inputHidden{}".format(i + 1),
                                                  trainable=True)
                        activation_ = inputHiddenLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{inputHiddenLayer.kernel.name}"], inputHiddenLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{inputHiddenLayer.bias.name}"], inputHiddenLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    inputHiddenOutput_ = tf.identity(activation_, name="inputHiddenOutput")
                    inputOutputLayer = tf.layers.Dense(units=self.numActionSpace, activation=tf.sigmoid,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="forgetOutputLayer{}".format(len(sigmoidLayersWidths) + 1),
                                                        trainable=True)
                    inputOutput_ = inputOutputLayer(inputHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{inputOutputLayer.kernel.name}"], inputOutputLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{inputOutputLayer.bias.name}"], inputOutputLayer.bias)
                    tf.add_to_collections("inputOutput", inputOutput_)


            with tf.variable_scope("tanhGate"):
                with tf.variable_scope("trainHiddenTanh"):
                    activation_ = inputStates_
                    for i in range(len(tanhLayerWidths)):
                        tanhHiddenLayer = tf.layers.Dense(units=tanhLayerWidths[i], activation=None,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="hiddenTanh{}".format(i + 1),
                                                  trainable=True)
                        activation_ = tanhHiddenLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{tanhHiddenLayer.kernel.name}"], tanhHiddenLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{tanhHiddenLayer.bias.name}"], tanhHiddenLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    tanhHiddenOutput_ = tf.identity(activation_, name="tanhHiddenOutput")
                    outputTanhLayer = tf.layers.Dense(units=self.numActionSpace, activation=tf.tanh,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="outputTanh{}".format(len(tanhLayerWidths) + 1),
                                                        trainable=True)
                    tanhOutput_ = outputTanhLayer(tanhHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{outputTanhLayer.kernel.name}"], outputTanhLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{outputTanhLayer.bias.name}"], outputTanhLayer.bias)
                    tf.add_to_collections("tanhOutput", tanhOutput_)

            with tf.variable_scope("opSigmoidGate"):
                with tf.variable_scope("trainOpHidden"):
                    activation_ = inputStates_
                    for i in range(len(sigmoidLayersWidths)):
                        opHiddenLayer = tf.layers.Dense(units=sigmoidLayersWidths[i], activation=None,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="opHidden{}".format(i + 1),
                                                  trainable=True)
                        activation_ = opHiddenLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{opHiddenLayer.kernel.name}"], opHiddenLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{opHiddenLayer.bias.name}"], opHiddenLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    opHiddenOutput_ = tf.identity(activation_, name="opHiddenOutput")
                    opOutputLayer = tf.layers.Dense(units=self.numActionSpace, activation=tf.sigmoid,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="opOutputLayer{}".format(len(sigmoidLayersWidths) + 1),
                                                        trainable=True)
                    opOutput_ = opOutputLayer(opHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{opOutputLayer.kernel.name}"], opOutputLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{opOutputLayer.bias.name}"], opOutputLayer.bias)
                    tf.add_to_collections("opOutput", opOutput_)

            with tf.variable_scope("trainingParams"):
                learningRate_ = tf.constant(0.001, dtype=tf.float32)
                tf.add_to_collection("learningRate", learningRate_)

            with tf.variable_scope("cell"):
                outputCell_ = forgetOutput_*formerCell_+inputOutput_*tanhOutput_
                tf.add_to_collection("outputCell", outputCell_)

            with tf.variable_scope("output"):
                output_ = tf.tanh(outputCell_)*opOutput_
                tf.add_to_collection("output", output_)

            with tf.variable_scope("QTable"):
                QEval_ = output_
                tf.add_to_collections("QEval", QEval_)
                # loss_ = tf.reduce_mean(tf.square(yi_ - QEval_))
                loss_ = tf.reduce_mean(tf.square(yi_ - QEval_))
                # loss_ = tf.losses.mean_squared_error(labels=yi_, predictions=QEval_)
                tf.add_to_collection("loss", loss_)

            with tf.variable_scope("train"):
                trainOpt_ = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer').minimize(loss_)
                tf.add_to_collection("trainOp", trainOpt_)

                saver = tf.train.Saver(max_to_keep=None)
                tf.add_to_collection("saver", saver)

            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)
            if summaryPath is not None:
                trainWriter = tf.summary.FileWriter(summaryPath + "/train", graph=tf.get_default_graph())
                testWriter = tf.summary.FileWriter(summaryPath + "/test", graph=tf.get_default_graph())
                tf.add_to_collection("writers", trainWriter)
                tf.add_to_collection("writers", testWriter)
            saver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", saver)

            # self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
            #         for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        return model


class TrainOneStep:

    def __init__(self, learningRate, actionDim, batchSize):
        self.learningRate = learningRate
        self.actionDim = actionDim
        self.formerCellBatch = np.random.rand(batchSize, actionDim)
        self.formerOutputBatch = np.random.rand(batchSize, actionDim)
        self. batchSize = batchSize

    def __call__(self, model, statesBatch, yBatch):

        # print("ENTER TRAIN")
        graph = model.graph
        yi_ = graph.get_collection_ref("yi")[0]
        learningRate_ = graph.get_collection_ref("learningRate")[0]
        loss_ = graph.get_collection_ref("loss")[0]
        trainOp_ = graph.get_collection_ref("trainOp")[0]
        output_ = graph.get_collection_ref('output')[0]
        outputCell_ = graph.get_collection_ref("outputCell")[0]
        states_ = graph.get_collection_ref("states")[0]
        formerCell_ = graph.get_collection_ref("formerCell")[0]
        formerOutput_ = graph.get_collection_ref("formerOutput")[0]
        fetches = [loss_, trainOp_]

        statesBatch = np.asarray(statesBatch).reshape(self.batchSize, -1)
        self.formerOutputBatch = np.asarray(self.formerOutputBatch).reshape(self.batchSize, -1)
        self.formerCellBatch = np.asarray(self.formerCellBatch).reshape(self.batchSize, -1)

        outputBatch = model.run(output_, feed_dict={states_: statesBatch, formerCell_: self.formerCellBatch,
                                                         formerOutput_: self.formerOutputBatch})
        cellBatch = model.run(outputCell_, feed_dict={states_: statesBatch, formerCell_: self.formerCellBatch,
                                                           formerOutput_: self.formerOutputBatch})
        feedDict = {states_: statesBatch, learningRate_: self.learningRate, yi_: yBatch, formerCell_: self.formerCellBatch,
                    formerOutput_: self.formerOutputBatch}
        lossDict, trainOp = model.run(fetches, feed_dict=feedDict)
        self.formerOutputBatch = outputBatch
        self.formerCellBatch = cellBatch
        return model, lossDict, self.formerCellBatch, self.formerOutputBatch


class RunAlgorithm:

    def __init__(self, learningRate, actionDim, batchSize, trainingRounds, trainOneStep):
        self.learningRate = learningRate
        self.actionDim = actionDim
        self.batchSize = batchSize
        self.trainingRounds = trainingRounds
        self.trainOneStep = trainOneStep

    def __call__(self, model, statesBatch, yBatch):
        LOSS = []
        ACC = []
        for _ in range(self.trainingRounds):
            model, loss, cell, output = self.trainOneStep(model, statesBatch, yBatch)
            accuracy = 0
            for i in range(len(yBatch)):
                if isIn(groundTruth[np.argmax(statesBatch[i])], np.argmax(output[i])):
                    accuracy += 1
            print("loss:{}, acc:{}".format(loss, accuracy/self.batchSize))
            LOSS.append(loss)
            ACC.append(accuracy/self.batchSize)

        return model, loss, cell, output, LOSS, ACC


def runTestSet(model, statesBatch, output, cell):
    graph = model.graph
    yi_ = graph.get_collection_ref("yi")[0]
    output_ = graph.get_collection_ref('output')[0]
    outputCell_ = graph.get_collection_ref("outputCell")[0]
    states_ = graph.get_collection_ref("states")[0]
    formerCell_ = graph.get_collection_ref("formerCell")[0]
    formerOutput_ = graph.get_collection_ref("formerOutput")[0]
    batchSize = len(statesBatch)
    statesBatch = np.asarray(statesBatch).reshape(batchSize, -1)
    output = np.asarray(output).reshape(batchSize, -1)
    cell = np.asarray(cell).reshape(batchSize, -1)

    result = model.run(output_, feed_dict={states_: statesBatch, formerOutput_: output, formerCell_: cell})
    accuracy = 0
    for i in range(len(statesBatch)):
        # if np.argmax(yBatch[i]) == np.argmax(result[i]):
        if i < batchSize - 1 and (groundTruth[np.argmax(statesBatch[i])], np.argmax(result[i])):
            accuracy += 1

    return accuracy / len(statesBatch)