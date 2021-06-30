from numpy import *  
import time  
import matplotlib.pyplot as plt   
from sklearn.metrics import accuracy_score

# 计算核函数的值
def calcKernelValue(matrix_x, sample_x, kernelOption):  
	kernelType = kernelOption[0]  
	numSamples = matrix_x.shape[0]  
	kernelValue = mat(zeros((numSamples, 1)))  

	if kernelType == 'linear':  
		kernelValue = matrix_x * sample_x.T  
	elif kernelType == 'rbf':  
		sigma = kernelOption[1]  
		if sigma == 0:  
			sigma = 1.0  
		for i in range(numSamples):  
			diff = matrix_x[i, :] - sample_x  
			kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma**2))  
	else:  
		raise NameError('Not support kernel type! You can use linear or rbf!')  
	return kernelValue   
  
  
# 利用核函数将数据集映射到高维空间
def calcKernelMatrix(train_x, kernelOption):  
	numSamples = train_x.shape[0]  
	kernelMatrix = mat(zeros((numSamples, numSamples)))  
	for i in range(numSamples):  
		kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)  
	return kernelMatrix  
  

# 定义SVM结构体
class SVMStruct:  
	def __init__(self, dataSet, labels, C, toler, kernelOption):  
		self.train_x = dataSet # each row stands for a sample  
		self.train_y = labels  # corresponding label  
		self.C = C             # slack variable(松弛变量系数)  
		self.toler = toler     # termination condition for iteration  
		self.numSamples = dataSet.shape[0] # number of samples  
		self.alphas = mat(zeros((self.numSamples, 1))) # Lagrange factors for all samples(拉格朗日系数，需要优化项)  
		self.b = 0  #阈值
		self.errorCache = mat(zeros((self.numSamples, 2)))  
		self.kernelOpt = kernelOption  #核选项，如果是线性核kernelOption=('linear', 0) 如果是高斯核kernelOption=('rbf', sigma)，sigma为高斯核参数
		self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)  
  

# 计算误差
def calcError(svm, alpha_k):  
	output_k = float(multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_k] + svm.b)  
	error_k = output_k - float(svm.train_y[alpha_k])  
	return error_k  

# 为了节省计算时间，我们建立一个全局的缓存用于保存所有样本的误差值，而不用每次选择的时候就重新计算。
# 计算并更新误差到缓存errorCache中
def updateError(svm, alpha_k):  
	error = calcError(svm, alpha_k)  
	svm.errorCache[alpha_k] = [1, error]  

# 启发式算法选择j，选择具有最大步长的j
def selectAlpha_j(svm, alpha_i, error_i):  
	svm.errorCache[alpha_i] = [1, error_i] # mark as valid(has been optimized)  	
	candidateAlphaList = nonzero(svm.errorCache[:, 0].A)[0] # mat.A return array  
	maxStep = 0; alpha_j = 0; error_j = 0  
    # 遍历缓存，选择具有最大步长（error_k - error_i）的alpha_k作为alpha_j
	if len(candidateAlphaList) > 1:  
		for alpha_k in candidateAlphaList:  
			if alpha_k == alpha_i:   
				continue  
			error_k = calcError(svm, alpha_k)  
			if abs(error_k - error_i) > maxStep:  
				maxStep = abs(error_k - error_i)  
				alpha_j = alpha_k  
				error_j = error_k  
	# 第一次循环，随机选择alpha j
	else:             
		alpha_j = alpha_i  
		while alpha_j == alpha_i:  
			alpha_j = int(random.uniform(0, svm.numSamples))  
		error_j = calcError(svm, alpha_j)  
      
	return alpha_j, error_j 
    
  
  
# 内循环：优化alpha_i，alpha_j
def innerLoop(svm, alpha_i):  
	error_i = calcError(svm, alpha_i)  
    ## 满足 KKT 条件  
    # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)  
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)  
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)  
    ## 违反 KKT 条件  
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)   
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)  
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized  
	if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):    
        # 1. 选择alpha j  
		alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)  
		alpha_i_old = svm.alphas[alpha_i].copy()  
		alpha_j_old = svm.alphas[alpha_j].copy()  
  
        # 2. 计算边界数据 alpha j 的 L 和 H
		if svm.train_y[alpha_i] != svm.train_y[alpha_j]:  
			L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])  
			H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])  
		else:  
			L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)  
			H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])  
		if L == H:  
			return 0  
  
        # 3. 计算 i 和 j 的相似性
		eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] - svm.kernelMat[alpha_j, alpha_j]  
		if eta >= 0:  
			return 0  
  
        # 4. 更新 alpha_j
		svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta  
  
        # 5. clip alpha j  
		if svm.alphas[alpha_j] > H:  
			svm.alphas[alpha_j] = H  
		if svm.alphas[alpha_j] < L:  
			svm.alphas[alpha_j] = L  
  
        # 6. 如果 alpha j 与上一次更新小于阈值，返回
		if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:  
			updateError(svm, alpha_j)  
			return 0  
  
        # 7. 更新 alpha i
		svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] * (alpha_j_old - svm.alphas[alpha_j])  
  
        # 8. 更新 b
		b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old)* svm.kernelMat[alpha_i, alpha_i] \
		*svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
		* svm.kernelMat[alpha_i, alpha_j]  
		b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
		* svm.kernelMat[alpha_i, alpha_j] \
		- svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
		* svm.kernelMat[alpha_j, alpha_j]  
		if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):  
			svm.b = b1  
		elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):  
			svm.b = b2  
		else:  
			svm.b = (b1 + b2) / 2.0  
  
        # 9. 更新 alpha_i， alpha_j
		updateError(svm, alpha_j)  
		updateError(svm, alpha_i)  
  
		return 1  
	else:  
		return 0  
  
  
# the main training procedure 
# train_x:数据集
# train_y：类别标签
# C:常数
# toler：容错率
# maxIter：退出前的循环次数 
def trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('rbf', 1.0)):  

	startTime = time.time() #记录训练时的开始时间 
	svm = SVMStruct(mat(train_x), mat(train_y), C, toler, kernelOption)  
      
    # 开始训练
	entireSet = True  
	alphaPairsChanged = 0  
	iterCount = 0  
    # 迭代终止条件：
	# 1. 达到最大迭代数
	# 2. alpha没有变化

	while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):  
		alphaPairsChanged = 0  
  
        # update alphas over all training examples  
		if entireSet:  
			for i in range(svm.numSamples):  
				alphaPairsChanged += innerLoop(svm, i)  
			print ('---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)  )
			iterCount += 1  
        # update alphas over examples where alpha is not 0 & not C (not on boundary)  
		else:  
			nonBoundAlphasList = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]  
			for i in nonBoundAlphasList:  
				alphaPairsChanged += innerLoop(svm, i)  
			print ('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)  )
			iterCount += 1  
  
        # alternate loop over all examples and non-boundary examples  
		if entireSet:  
			entireSet = False  
		elif alphaPairsChanged == 0:  
			entireSet = True  
  
	print ('Congratulations, training complete! Took %fs!' % (time.time() - startTime))  
	return svm  
  
  
# 测试函数
def testSVM(svm, test_x, test_y):  
	test_x = mat(test_x)  
	test_y = mat(test_y)  
	numTestSamples = test_x.shape[0]  
	supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]  
	supportVectors      = svm.train_x[supportVectorsIndex]  
	supportVectorLabels = svm.train_y[supportVectorsIndex]  
	supportVectorAlphas = svm.alphas[supportVectorsIndex]  
	matchCount = 0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for i in range(numTestSamples):  
		kernelValue = calcKernelValue(supportVectors, test_x[i, :], svm.kernelOpt)  
		predict = kernelValue.T * multiply(supportVectorLabels, supportVectorAlphas) + svm.b  
		if sign(predict) == sign(test_y[i]): 
			if(sign(test_y[i])>=0):
				TP += 1  
			else:
				TN += 1
		else:
			if(sign(test_y[i])>=0):
				FP += 1  
			else:
				FN += 1
		
	Accuracy = float(TP+TN+0.1) / (TP+TN+FP+FN+0.1)
	Precision = float(TP+0.1) / (TP+FP+0.1)
	Recall = float(TP+0.1) / (TP+FN+0.1)
	F1score = (2*Precision*Recall+0.1) / (Precision+Recall+0.1)
	return Accuracy, Precision, Recall,F1score