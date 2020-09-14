import random
import pandas as pd
import numpy as np

table = "BTPSXVE"

def reberGrammar(n):
    graph = {"B": ["T1","P1"],
             "T1": ["S1","X1"],
             "P1": ["T1","V1"],
             "S1": ["S1","X1"],
             "T2": ["T2","V1"],
             "X1": ["X2","S2"],
             "X2": ["T2","V1"],
             "V1": ["P2","V2"],
             "P2": ["X2","S2"],
             "S2": ["E"],
             "V2": ["E"],
             "E": ["end"]}
    rebers = []
    for i in range(n):
        strI = ""
        edge_ = "B"
        while edge_ != "end":
            strI += edge_[0]
            subEdge_ = graph[edge_]
            edge_ = random.sample(subEdge_,1)[0]
        rebers.append(strI)
    return rebers


def embeddedReberGrammar(rebers):
    newRebers=[]
    for _,str_ in enumerate(rebers):
        type = random.randint(0,1)
        if type == 0:
            newRebers.append("BT"+str_+"TE")
        else:
            newRebers.append("BP"+str_+"PE")
    return newRebers


def translateReberToNumber(rebers):
    xData = []
    yData = []
    strX = ""
    strY = ""
    for i in range(len(rebers)):
        for j in range(len(rebers[i])-1):
            strX += str(table.index(rebers[i][j]))
            strY += str(table.index(rebers[i][j+1]))
    for i in range(len(strX)):
        xData.append([int(strX[i])])
        yData.append([int(strY[i])])
    return xData, yData


def transformToOneHot(xData, yData, stateDim):
    oneHotState = []
    oneHotY = []
    for i in range(len(xData)):
        init = np.zeros(stateDim)
        init[xData[i]] = 1
        oneHotState.append([init])
        init = np.zeros(stateDim)
        init[yData[i]] = 1
        oneHotY.append(init)
    return oneHotState, oneHotY


def isIn(list, number):
    for i in range(len(list)):
        if list[i] == number:
            return 1

    return 0


