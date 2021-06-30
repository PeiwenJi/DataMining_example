from numpy import *  
import numpy as np
import svm
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


################## test svm #####################  
## step 1: load data  
print ("step 1: load data...")
data = []  
dataSet = []  
labels = [] 
accuracy = [] 
precision = []
recall = []
F1score = []
f = open('./heart_failure_clinical_records_dataset.txt')
raw_list = f.readlines()
random.shuffle(raw_list)
for line in raw_list:  
	lineArr = line.strip().split('\t')  
	dataSet.append([float(lineArr[0]), float(lineArr[1]),float(lineArr[2]),float(lineArr[3]),float(lineArr[4]),float(lineArr[5]),float(lineArr[6]),float(lineArr[7]),float(lineArr[8]),float(lineArr[9]),float(lineArr[10]),float(lineArr[11])])  
	labels.append(float(lineArr[12]))
#将数据转化为矩阵，并将labels转置一下,于是类别标签向量的每行元素都和数据矩阵中的行一一对应。
labels = mat(labels).T 
dataSet = mat(dataSet)

## step 2: training...  
print ("step 2: training..." ) 
# 模型参数
C = 1
toler = 0.001  
maxIter = 50
# 十字交叉
for i in range(0,271,30):
	# 测试数据
	test_x = dataSet[i:i+31, :]  
	test_y = labels[i:i+31, :] 
	# 训练数据
	train_x1 = dataSet[0:i+1, :]  
	train_y1 = labels[0:i+1, :]
	train_x2 = dataSet[i+30:301, :]  
	train_y2 = labels[i+30:301, :]
	train_x = np.append(train_x1, train_x2, axis=0)
	train_y = np.append(train_y1, train_y2, axis=0)
	# 训练模型
	svmClassifier = svm.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption =  ('linear', 0))   
	# 评估模型
	evaluation = svm.testSVM(svmClassifier, test_x, test_y)
	# 取得评估参数：accuracy, precision, recall, F1score
	accuracy.append(evaluation[0])
	precision.append(evaluation[1])
	recall.append(evaluation[2])
	F1score.append(evaluation[3])

## step 3: testing 
print ("step 3: printing..." )
# 控制台打印各项信息
print ("accuracy:" )
print (np.mean(accuracy))
print ("precision:" )
print (np.mean(precision))
print ("recall:" ) 
print (np.mean(recall))
print ("F1score:" )
print (np.mean(F1score))

