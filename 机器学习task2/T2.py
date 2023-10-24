from torchvision import datasets, transforms
import torch.utils.data
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

from joblib import dump
from joblib import load

import torch.nn as nn
import torch.optim as optim

# 数据准备
train_data = datasets.MNIST(
                            root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=False,
                            )

test_data = datasets.MNIST(
                            root='./data',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=False,
                            )
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=1,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=1,
                                          shuffle=None)

# sigmoid激活函数
def sigmoid(z):
    # for r in range(z.shape[0]):
    #     for c in range(z.shape[1]):
    #         if z[r,c] >= 0:
    #             z[r,c] = 1 / (1 + np.exp(-z[r,c]))
    #         else :
    #             z[r,c] = np.exp(z[r,c]) / (1 + np.exp(z[r,c]))
    # return np.exp(z) / (1 + np.exp(z))
    for r in range(z.shape[0]):
        for c in range(z.shape[1]):
            if z[r, c] >= 0:
                z[r, c] = 1 / (1 + np.exp(-z[r, c]))
            else:
                z[r, c] = np.exp(z[r, c]) / (1 + np.exp(z[r, c]))
    return z


def cost(prediction, labels):
    return np.mean(np.power(prediction - labels, 2))

def loadImageData(train_loader, test_loader):

    trainDatas = np.empty(shape=(0, 784))
    trainLabels = np.empty(shape=(0, 1))
    number = 0
    for data, label in train_loader:
        number += 1
        data = np.array(data)
        for x in data:
            data_ = x.reshape((1, 784))
        # data_ = data.reshape((1, 784))
        trainLabels = np.append(trainLabels, np.array(label))
        trainDatas = np.append(trainDatas, data_, axis=0)
        print("训练集已经处理了", number, "个数据", "------", number, "/60000")
        if number==10000:
            break
        # print(type(np.array(label)))
        #
        # break
    testDatas = np.empty(shape=(0, 784))
    testLabels = np.empty(shape=(0, 1))
    number_ = 0
    for data, label in test_loader:
        number_ += 1
        data = np.array(data)
        for x in data:
            data_ = x.reshape((1, 784))
        # data = data.reshape((1, 784))
        testLabels = np.append(testLabels, np.array(label))
        testDatas = np.append(testDatas, data_, axis=0)
        print("测试集已经处理了", number_, "个数据", "------", number_, "/10000")
        if number_==1667:
            break

    tmean = np.mean(trainDatas)
    tstd = np.std(testDatas)

    trainDatas = (trainDatas-tmean) / tstd
    testDatas = (testDatas-tmean) / tstd

    return trainDatas, trainLabels, testDatas, testLabels


# 对输出标签进行OneHot编码  参数：labels 待编码的标签  Label_class  编码类数
def OneHotEncoder(labels, Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
    return one_hot_label


# 训练一轮ANN 参数：训练数据 标签 输入层 隐层 输出层size 输入层隐层连接权重 隐层输出层连接权重 偏置1  偏置2 学习率
def trainANN(X, y, input_size, hidden_size, output_size, omega1, omega2, theta1, theta2, learningRate):
    # 获取样本个数
    m = X.shape[0]
    # 将矩阵X,y转换为numpy型矩阵
    X = np.matrix(X)
    y = np.matrix(y)

    # 前向传播 计算各层输出
    # 隐层输入 shape=m*hidden_size
    h_in = np.matmul(X, omega1.T) - theta1.T
    # 隐层输出 shape=m*hidden_size
    h_out = sigmoid(h_in)
    # 输出层的输入 shape=m*output_size
    o_in = np.matmul(h_out, omega2.T) - theta2.T
    # 输出层的输出 shape=m*output_size
    o_out = sigmoid(o_in)

    # 当前损失
    all_cost = cost(o_out, y)

    # 反向传播
    # 输出层参数更新
    d2 = np.multiply(np.multiply(o_out, (1 - o_out)), (y - o_out))
    omega2 += learningRate * np.matmul(d2.T, h_out) / m
    theta2 -= learningRate * np.sum(d2.T, axis=1) / m

    # 隐层参数更新
    d1 = np.multiply(h_out, (1 - h_out))
    omega1 += learningRate * (np.matmul(np.multiply(d1, np.matmul(d2, omega2)).T, X) / float(m))
    theta1 -= learningRate * (np.sum(np.multiply(d1, np.matmul(d2, omega2)).T, axis=1) / float(m))

    return omega1, omega2, theta1, theta2, all_cost


# 数据预测
def predictionANN(X, omega1, omega2, theta1, theta2):
    # 获取样本个数
    m = X.shape[0]
    # 将矩阵X,y转换为numpy型矩阵
    X = np.matrix(X)

    # 前向传播 计算各层输出
    # 隐层输入 shape=m*hidden_size
    h_in = np.matmul(X, omega1.T) - theta1.T
    # 隐层输出 shape=m*hidden_size
    h_out = sigmoid(h_in)
    # 输出层的输入 shape=m*output_size
    o_in = np.matmul(h_out, omega2.T) - theta2.T
    # 输出层的输出 shape=m*output_size
    o_out = np.argmax(sigmoid(o_in), axis=1)

    return o_out


# 计算模型准确率
def computeAcc(X, y, omega1, omega2, theta1, theta2):
    y_hat = predictionANN(X, omega1, omega2, theta1, theta2)
    return np.mean(y_hat == y)

if __name__ == '__main__':
    trainData, trainLabels, testData, testLabels = loadImageData(train_loader=train_loader, test_loader=test_loader)

    # 初始化设置
    input_size = 784
    hidden_size = 64
    output_size = 10
    lamda = 1

    # 将网络参数进行随机初始化
    omega1 = np.matrix((np.random.random(size=(hidden_size, input_size)) - 0.5) * 0.25)  # 15*784
    omega2 = np.matrix((np.random.random(size=(output_size, hidden_size)) - 0.5) * 0.25)  # 10*15

# 初始化两个偏置
    theta1 = np.matrix((np.random.random(size=(hidden_size,1)) - 0.5) * 0.25) # 15*1
    theta2 = np.matrix((np.random.random(size=(output_size,1)) - 0.5) * 0.25) # 10*1
    # 学习率
    learningRate = 0.1
    # 数据集
    m = trainData.shape[0] # 样本个数
    X = np.matrix(trainData) # 输入数据 m*784
    y_onehot = OneHotEncoder(trainLabels, 10)  # 标签 m*10

    iters_num = 20000  # 设定循环的次数
    loss_list = []
    acc_list = []
    acc_max = 0.0  # 最大精度 在精度达到最大时保存模型
    acc_max_iters = 0

    for i in range(iters_num):
        omega1, omega2, theta1, theta2, loss = trainANN(X, y_onehot, input_size, hidden_size, output_size, omega1,
                                                        omega2, theta1, theta2, learningRate)
        loss_list.append(loss)
        acc_now = computeAcc(testData, testLabels, omega1, omega2, theta1, theta2)  # 计算精度
        acc_list.append(acc_now)
        if acc_now > acc_max:  # 如果精度达到最大 保存模型
            acc_max = acc_now
            acc_max_iters = i  # 保存坐标 方便在图上标注
            # 保存模型参数
            f = open(r"../招新/best_model", 'wb')
            pickle.dump((omega1, omega2, theta1, theta2), f, 0)
            f.close()
        print("%d  Now accuracy : %f" % (i, acc_now))

        # if i % 100 == 0:  # 每训练100轮打印一次精度信息
        #     print("%d  Now accuracy : %f" % (i, acc_now))

    # 保存训练数据 方便分析
    f = open(r"./loss_list", 'wb')
    pickle.dump(loss_list, f, 0)
    f.close()

    f = open(r"./acc_list", 'wb')
    pickle.dump(loss_list, f, 0)
    f.close()

    # 绘制图形
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    x1 = np.arange(len(loss_list))
    plt.plot(x1, loss_list, "r")
    plt.xlabel(r"Number of iterations", fontsize=16)
    plt.ylabel(r"Mean square error", fontsize=16)
    plt.grid(True, which='both')

    plt.subplot(122)
    x2 = np.arange(len(acc_list))
    plt.plot(x2, acc_list, "r")
    plt.xlabel(r"Number of iterations", fontsize=16)
    plt.ylabel(r"Accuracy", fontsize=16)
    plt.grid(True, which='both')
    plt.annotate('Max accuracy:%f' % (acc_max),  # 标注最大精度值
                 xy=(acc_max_iters, acc_max),
                 xytext=(acc_max_iters * 0.7, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 ha="center",
                 fontsize=15,
                 )
    plt.plot(np.linspace(acc_max_iters, acc_max_iters, 200), np.linspace(0, 1, 200), "y--", linewidth=2, )  # 最大精度迭代次数
    plt.plot(np.linspace(0, len(acc_list), 200), np.linspace(acc_max, acc_max, 200), "y--", linewidth=2)  # 最大精度

    plt.scatter(acc_max_iters, acc_max, s=180, facecolors='#FFAAAA')  # 标注最大精度点
    plt.axis([0, len(acc_list), 0, 1.0])  # 设置坐标范围
    plt.savefig("ANN_plot")  # 保存图片
    plt.show()









