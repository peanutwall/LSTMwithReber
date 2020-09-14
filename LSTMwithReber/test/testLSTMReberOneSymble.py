import tensorflow as tf
import numpy as np
import random
from collections import deque
import os
from generateReberStr import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

            with tf.variable_scope("sigmoidGate"):
                with tf.variable_scope("trainHiddenSigmoid"):
                    activation_ = inputStates_
                    for i in range(len(sigmoidLayersWidths)):
                        sigmoidHiddenLayer = tf.layers.Dense(units=sigmoidLayersWidths[i], activation=None,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="hiddenSigmoid{}".format(i + 1),
                                                  trainable=True)
                        activation_ = sigmoidHiddenLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{sigmoidHiddenLayer.kernel.name}"], sigmoidHiddenLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{sigmoidHiddenLayer.bias.name}"], sigmoidHiddenLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    sigmoidHiddenOutput_ = tf.identity(activation_, name="sigmoidHiddenOutput")
                    sigmoidOutputLayer = tf.layers.Dense(units=self.numActionSpace, activation=tf.sigmoid,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="outputSigmoid{}".format(len(sigmoidLayersWidths) + 1),
                                                        trainable=True)
                    sigmoidOutput_ = sigmoidOutputLayer(sigmoidHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{sigmoidOutputLayer.kernel.name}"], sigmoidOutputLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{sigmoidOutputLayer.bias.name}"], sigmoidOutputLayer.bias)
                    tf.add_to_collections("sigmoidOutput", sigmoidOutput_)

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

            with tf.variable_scope("trainingParams"):
                learningRate_ = tf.constant(0.001, dtype=tf.float32)
                tf.add_to_collection("learningRate", learningRate_)

            with tf.variable_scope("cell"):
                outputCell_ = sigmoidOutput_*formerCell_+sigmoidOutput_*tanhOutput_
                tf.add_to_collection("outputCell", outputCell_)

            with tf.variable_scope("output"):
                output_ = tf.tanh(outputCell_)*sigmoidOutput_
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

    def __init__(self, learningRate, actionDim, batchSize=1):
        self.learningRate = learningRate
        self.actionDim = actionDim
        self.formerCellBatch = np.random.rand(batchSize, actionDim)
        self.formerOutputBatch = np.random.rand(batchSize, actionDim)

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

        statesBatch = np.asarray(statesBatch).reshape(batchSize, -1)
        self.formerOutputBatch = np.asarray(self.formerOutputBatch).reshape(batchSize, -1)
        self.formerCellBatch = np.asarray(self.formerCellBatch).reshape(batchSize, -1)

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


stateDim = 7
actionDim = 7
learningRate = 0.1
numStrings = 256
layerWidths = [7]
buildModel = BuildModel(stateDim, actionDim)
rebersTrain = embeddedReberGrammar(reberGrammar(numStrings))
batchSize = 1
trainOneStep = TrainOneStep(learningRate, actionDim, batchSize)
model = buildModel(layerWidths, layerWidths)
testRounds = 500
accuracy = 0
for l in range(testRounds):
    for i in range(numStrings):
        xData, yData = translateReberToNumber([rebersTrain[i]])
        statesBatch, yBatch = transformToOneHot(xData, yData, stateDim)
        for j in range(len(statesBatch)):
            model, loss, cell, output = trainOneStep(model, statesBatch[j], [yBatch[j]])
            if l == testRounds-1 and np.argmax(yBatch[j]) == np.argmax(output):
                accuracy += 1

print(accuracy)
print(accuracy/len(translateReberToNumber(rebersTrain)[0]))



