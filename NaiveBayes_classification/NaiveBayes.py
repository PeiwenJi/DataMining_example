# coding=utf-8
import csv
import random
import math
import numpy as np
import pandas as pd
from math import log
from collections import Counter
from pandas.core.frame import DataFrame


# 加载数据集
def loadData(path):
        dataSet=[]
        f = open(path)
        # 将数据集随机打散
        raw_list = f.readlines()
        random.shuffle(raw_list)
        for line in raw_list:
                # 用逗号分隔文本
                lineArr = line.strip().split(',')  
                dataSet.append([float(lineArr[0]), float(lineArr[1]),float(lineArr[2]),float(lineArr[3])]) 
        # print(dataSet)
        print("Load dataset successfully!")
        return dataSet 

# 计算某一列的信息熵
def calcShannonEnt(dataSet,axis):                      
    numEntires = len(dataSet) 
    columnCounter = Counter(dataSet[:,axis])
    shannonEnt = 0.0                                
    for key in columnCounter.keys():                       
        prob = float(columnCounter[key]) / numEntires  
        shannonEnt -= prob * log(prob, 2)           
    return shannonEnt                           

# 以信息增益作为标准，对各个属性的值进行离散化，离散化成bool类型
def split(dataSet, axis):
    #inf为正无穷大
    minEntropy = np.inf  
    #记录最终分割索引
    index = -1  
    #按第axis列对数据进行排序
    sortData = dataSet[np.argsort(dataSet[:,axis])]
    #初始化最终分割数据后的熵
    lastE1,lastE2 = -1,-1
    for i in range(len(sortData)):
        #分割数据集
        splitData1,splitData2 = sortData[: i + 1],sortData[i + 1 :]
        entropy1,entropy2 = (
            calcShannonEnt(splitData1,axis),
            calcShannonEnt(splitData2,axis),
        ) #计算信息熵
        entropy = entropy1 * len(splitData1) / len(sortData) + \
                entropy2 * len(splitData2) / len(sortData)
        #如果调和平均熵小于最小值
        if entropy < minEntropy:
            minEntropy = entropy
            index = i
            lastE1 = entropy1
            lastE2 = entropy2
    print(("Threshold is {0} in {1} column").format(sortData[index,axis], axis))
    sortData[: index + 1,axis] = 0
    sortData[index + 1 :,axis] = 1
    print(("Discretize {0} column successfully!").format(axis))
    return sortData

# 以splitRatio比例将数据集分成训练集和测试集
def splitDataset(dataset, splitRatio):
        trainSize = int(len(dataset) * splitRatio)
        trainSet = []
        copy = list(dataset)
        while len(trainSet) < trainSize:
                index = random.randrange(len(copy))
                trainSet.append(copy.pop(index))
        print(('Split {0} rows into train with {1} rows and test with {2} rows').format(len(dataset), len(trainSet), len(copy)))
        return [trainSet, copy]

# 朴素贝叶斯模型
class NaiveBayes:
    def __init__(self):
        self.model = {}#key 为类别名 val 为字典PClass表示该类的该类，PFeature:{}对应对于各个特征的概率
    def calEntropy(self, y): # 计算熵
        valRate = y.value_counts().apply(lambda x : x / y.size) # 频次汇总 得到各个特征对应的概率
        valEntropy = np.inner(valRate, np.log2(valRate)) * -1
        return valEntropy
 
    def fit(self, xTrain, yTrain = pd.Series()):
        if not yTrain.empty:#如果不传，自动选择最后一列作为分类标签
            xTrain = pd.concat([xTrain, yTrain], axis=1)
        self.model = self.buildNaiveBayes(xTrain) 
        return self.model
    def buildNaiveBayes(self, xTrain):
        yTrain = xTrain.iloc[:,-1]
        
        yTrainCounts = yTrain.value_counts()# 频次汇总 得到各个特征对应的概率
 
        yTrainCounts = yTrainCounts.apply(lambda x : (x + 1) / (yTrain.size + yTrainCounts.size)) #使用了拉普拉斯平滑
        retModel = {}
        for nameClass, val in yTrainCounts.items():
            retModel[nameClass] = {'PClass': val, 'PFeature':{}}
 
        propNamesAll = xTrain.columns[:-1]
        allPropByFeature = {}
        for nameFeature in propNamesAll:
            allPropByFeature[nameFeature] = list(xTrain[nameFeature].value_counts().index)
        #print(allPropByFeature)
        for nameClass, group in xTrain.groupby(xTrain.columns[-1]):
            for nameFeature in propNamesAll:
                eachClassPFeature = {}
                propDatas = group[nameFeature]
                propClassSummary = propDatas.value_counts()# 频次汇总 得到各个特征对应的概率
                for propName in allPropByFeature[nameFeature]:
                    if not propClassSummary.get(propName):
                        propClassSummary[propName] = 0#如果有属性灭有，那么自动补0
                Ni = len(allPropByFeature[nameFeature])
                propClassSummary = propClassSummary.apply(lambda x : (x + 1) / (propDatas.size + Ni))#使用了拉普拉斯平滑
                for nameFeatureProp, valP in propClassSummary.items():
                    eachClassPFeature[nameFeatureProp] = valP
                retModel[nameClass]['PFeature'][nameFeature] = eachClassPFeature
 
        return retModel
    def predictBySeries(self, data):
        curMaxRate = None
        curClassSelect = None
        for nameClass, infoModel in self.model.items():
            rate = 0
            rate += np.log(infoModel['PClass'])
            PFeature = infoModel['PFeature']
            
            for nameFeature, val in data.items():
                propsRate = PFeature.get(nameFeature)
                if not propsRate:
                    continue
                rate += np.log(propsRate.get(val, 0))#使用log加法避免很小的小数连续乘，接近零
                #print(nameFeature, val, propsRate.get(val, 0))
            #print(nameClass, rate)
            if curMaxRate == None or rate > curMaxRate:
                curMaxRate = rate
                curClassSelect = nameClass
            
        return curClassSelect
    def predict(self, data):
        if isinstance(data, pd.Series):
            return self.predictBySeries(data)
        return data.apply(lambda d: self.predictBySeries(d), axis=1)


path = './titanic.txt'

print("step1: Loading dataSet...")
dataset = loadData(path)
dataset = np.array(dataset)

print("step2: Discretizing continuous attributes...")
dataset = split(dataset,0)
dataset = split(dataset,1)
dataset = split(dataset,2)

print("step3: Split dataSet into trainSet and testSet...")
splitRatio = 0.7
trainSet, testSet = splitDataset(dataset, splitRatio)

print("step4: Train model...")
trainSet = DataFrame(trainSet)
testSet = DataFrame(testSet)
naiveBayes = NaiveBayes()
treeData = naiveBayes.fit(trainSet)
 
print("step5: Test model...")
pd = pd.DataFrame({'预测值':naiveBayes.predict(testSet), '真实值':testSet.iloc[:,-1]})
print(pd)
print('正确率:%f%%'%(pd[pd['预测值'] == pd['真实值']].shape[0] * 100.0 / pd.shape[0]))

