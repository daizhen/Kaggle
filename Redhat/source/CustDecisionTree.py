from DecisionUtil import  *
from DecisionTreeNode import *
import numpy as np


class CustDecisionTreeClassifier:

    def __init__(self, columnNames=None):
        self.columnNames = columnNames
        self.tree = None
        self.X = None
        self.y=None
        self.buildQueue=[]

    def fit(self,X,y):
        self.X = X
        self.y = y
        self.__buildTree()

    def predict(self,X):
        predictResult = []
        for item in X:
            predictResult.append(self.predict(item))
        return predictResult;

    def showTree(self):
        self.__showNode(self.tree,"")

    def __predict(self,dataX,treeNode):
        if treeNode.results != None:
            return dataX.results
        col = treeNode.col
        colValue = treeNode.value
        if self.__meetCondition(dataX,col,colValue):
            return self.__predict(dataX,treeNode.tb);
        else:
            return self.__predict(dataX, treeNode.fb);

    def __showNode(self, treeNode, indent):
        if treeNode.results != None:
            print indent + ":"+str(treeNode.results)
        else :
            print indent + "IF: " + str(treeNode.col) + " " + str(treeNode.value)
            self.__showNode(treeNode.tb, indent + '\t');
            print indent + "ELSE:"
            self.__showNode(treeNode.fb, indent + '\t')

    def __meetCondition(self,X, col, value):
        if isinstance(value,int) or isinstance(value, float):
            return X[col] >=value
        else:
            return X[col] == value

    def __split(self, X, col, value):
        dataSet1 = None
        dataSet2 = None
        if isinstance(value,int) or isinstance(value,float):
            dataSet1 = X[:,col] >=value
            dataSet2 = X[:, col] < value
        else:
            dataSet1 = X[:, col] == value
            dataSet2 = X[:, col] != value
        range = np.arange(len(X))
        return range[dataSet1],range[dataSet2]


    def __buildTree(self):
        self.tree = self.__buildTreeNode(range(len(self.X)))
        while len(self.buildQueue) >0:
            buildNode = self.buildQueue.pop(0)
            newNode = self.__buildTreeNode(buildNode["dataSet"],buildNode["scoref"])
            if buildNode["branch"] == 't':
                buildNode["parent"].tb = newNode
            else:
                buildNode["parent"].fb = newNode

    def __pushNode(self,parentNode, data,scoref,branch):
        self.buildQueue.append({"parent":parentNode,"dataSet":data,"scoref":scoref,"branch":branch});

    def __buildTreeNode(self,dataSetIndexes, scoref = entropy):
        if len(dataSetIndexes) ==0:
            return DecisionTreeNode()
        current_score = scoref(self.y[dataSetIndexes])
        bestSplit_1 = None
        bestSplit_2 = None
        bestGain = 0
        bestCol = -1
        bestColValue = None
        column_count = len(self.X[0])
        for columnIndex in range(column_count):
            column_values = {}
            for rowIndex in range(len(self.X)):
                column_values[self.X[rowIndex,columnIndex]] = 1
            for column_value in column_values:
                (set1,set2) = self.__split(self.X[dataSetIndexes,:],columnIndex,column_value)
                #Calculate the infomation gain
                p1 = float(len(set1))/len(dataSetIndexes)
                gain = current_score - (p1*scoref(self.y[set1])+ (1-p1)*scoref(self.y[set2]))
                if gain > bestGain:
                    bestGain = gain
                    bestSplit_1 = set1
                    bestSplit_2 = set2
                    bestCol = columnIndex
                    bestColValue =column_value
        if bestGain > 0:
            #tb = self.__buildTree(bestSplit_1, scoref)
            #fb = self.__buildTree(bestSplit_2, scoref)
            if len(bestSplit_1) == len(dataSetIndexes) or len(bestSplit_2) == len(dataSetIndexes):
                print "error....."
            if len(bestSplit_1) == 0 or len(bestSplit_2) == 0:
                print "error....."
            newNode = DecisionTreeNode(col=bestCol,value=bestColValue)
            self.__pushNode(newNode, bestSplit_1, scoref,'t')
            self.__pushNode(newNode, bestSplit_2, scoref,'f')
            return newNode
        else:
            return DecisionTreeNode(results=uniqueCounts(self.y[dataSetIndexes]))

