from collections import Counter
import pandas as pd
import jieba
import json
import math
from sklearn.metrics import classification_report

# 10折交叉验证划分数据集和验证集
def cross_fold(k, f_path, ftest_path, ftrain_path):
    with open(f_path, 'r', encoding='UTF-8') as f:
        ftest = open(ftest_path, 'w+', encoding='UTF-8')
        ftrain = open(ftrain_path, 'w+', encoding='UTF-8')
        i = 0
        for line in f.readlines():
            if i % 10 == k:
                ftest.writelines(line)
            else:
                ftrain.writelines(line)
            i = i + 1
    ftest.close()
    ftrain.close()
    with open(ftrain_path, 'r', encoding='utf-8') as f_train:
        train = pd.read_table(f_train, names=['label', 'contents'])
    with open(ftest_path, 'r', encoding='utf-8') as f_test:
        test = pd.read_table(f_test, names=['label', 'contents'])
    train_x = train['contents']
    train_y = train['label']
    test_x = test['contents']
    test_y = test['label']
    return train_x, train_y, test_x, test_y


# jieba分词
def cut_words(dataset, stop_words_path):
    txt_list = [[] for _ in range(10)]
    with open(stop_words_path, 'r', encoding='utf-8') as fstop:
        stop_words_list = fstop.readlines()
        stop_words = [m.strip() for m in stop_words_list]
    j = 0
    for i in range(0, len(dataset)):
        word_bag = jieba.cut(dataset[i])
        for word in word_bag:
            if word not in stop_words and word != ' ':
                txt_list[j].append(word)
        if (i + 1) % 90 == 0:
            j += 1
    return txt_list

# 计算条件概率
def conditional_probability(dataset):
    keys = []
    for i in range(0, 10):
        keys.extend(list(dataset[i].keys()))
    keys_list = list(set(keys))
    category = [0] * 10
    for i in range(0, 10):
        for dic in dataset[i]:
            category[i] += dataset[i][dic]
    condition_pro = [{} for _ in range(0, 10)]
    for m in range(0, len(dataset)):
        for n in range(0, len(keys_list)):
            if keys_list[n] in dataset[m]:
                condition_pro[m].update({keys_list[n]: math.log((dataset[m][keys_list[n]] + 1) / (len(keys_list) + category[m]))})
            else:
                condition_pro[m].update({keys_list[n]: math.log(1 / (len(keys_list) + category[m]))})
    return condition_pro

# 预测每个测试样本的类别
def predict_label(c_p, t):
    posterior_probability = [{} for _ in range(len(t))]
    word_bags = [[] for _ in range(len(t))]
    tag = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
    predict_tag = []
    for i in range(len(t)):
        word_bags[i] = list(jieba.cut(t[i]))
        for j in range(10):
            pro = 1
            for k in dict(Counter(word_bags[i])).keys():
                if k in c_p[j]:
                    for number in range(Counter(word_bags[i])[k]):
                        pro *= c_p[j][k]
            posterior_probability[i].update({tag[j]: pro})
        predict_tag.append(max(posterior_probability[i], key=posterior_probability[i].get))
    return predict_tag

def main(k, f_path, ftest_path, ftrain_path, stop_words_path):
    #预处理
    print("step1-{0}: 划分数据集开始。一共有10份，这是第{0}份。".format(k+1))
    train_x, train_y, test_x, test_y = cross_fold(k, f_path, ftest_path, ftrain_path)
    print("step1-{0}: 划分数据集结束。".format(k+1))

    #jieba分词
    print("step2-{0}: jieba分词开始。".format(k+1))
    tag = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
    train_data = cut_words(train_x, stop_words_path)
    print("step2-{0}: jieba分词结束。".format(k+1))
    
    # 开始训练
    # 计算条件概率
    print("step3-{0}: 训练模型开始。".format(k+1))
    count = [[] for _ in range(10)]
    for i in range(0, len(train_data)):
        count[i] = Counter(train_data[i])
    con_pro = conditional_probability(count)
    print("step3-{0}: 训练模型结束。".format(k+1))

    # 开始测试
    print("step4-{0}: 测试开始。".format(k+1))
    test_tag = predict_label(con_pro, test_x)
    class_report = classification_report(test_y, test_tag)
    sum_of_correct = 0
    for i in range(0, len(train_data)):
        if(test_y[i] == test_tag[i]):
            sum_of_correct += 1
    accuracy = sum_of_correct/len(train_data)
    print("step4-{0}: 测试结束。".format(k+1))
    return class_report, test_y, test_tag, accuracy

if __name__ == '__main__':
    m = [[] for row in range(10)]
    testy = []
    predicty = [] 
    accuracy = 0
    for k in range(10):
        m[k], test, predict, accuracy = main(k, 'rawdata\cnews.txt', 'data\cnewstest.txt', 'data\cnewstrain.txt', 'rawdata\cnews.vocab.txt')
        testy.extend(test)
        predicty.extend(predict)
    report = classification_report(testy, predicty)
    print(report)
    print("准确率为：{0}".format(accuracy))
