'''
KNN算法--预测电影类别
'''
import numpy as np
import pandas as pb
import matplotlib.pyplot as plt

#函数作用为创造训练集数据（人为创造，函数只负责储存）
def TrainDateSet():
    '''
    group:特征数据集
    laberls:标签
    :return: group , laberls
    '''
    group = np.array([[1,108],[5,85],[10,36],[40,10],
                      [110,80],[80,110],[108,5],[115,8]])
    laberls = ['A','A','B','B','C','C','D','D']


    # 对group可视化处理（如下）
    print(group)
    num = 0
    X = []
    y = []
    while num<group.shape[0]:
       X.append(group[num][0])
       y.append(group[num][1])
       num = num + 1

    # print(X)
    # print(y)

    plt.scatter(X,y)
    plt.show()

    return group,laberls

#函数作用为创造测试集数据（人为创造，函数只负责储存）
def TestDateSet():
    '''
    Tgroup:测试数据
    :return: Tgroup
    '''
    Tgroup = np.array([[37,120],[15,80],[100,20],[120,6],
              [60,40],[120,37],[80,15]])

    num = 0
    X = []
    y = []
    while num<Tgroup.shape[0]:
        X.append(Tgroup[num][0])
        y.append(Tgroup[num][1])
        num = num + 1

    plt.scatter(X,y)
    plt.show()

    return Tgroup

#函数作用实行KNN算法
def KNN(group,laberls,TestDateSetGroup,num):
    KNNmatrix = []
    KNNmatrixNew = []
    # KNNnum = num
    for i in TestDateSetGroup:

        #构造对于一组测试元素关于训练数据集的欧式距离集合
        datacopy = np.tile(i,(group.shape[0],1))
        temp = np.subtract(datacopy,group)
        temp = np.power(temp,2)
        temp = np.array(temp)
        temp = temp.sum(axis=1)#注意

        sortedDistIndices = temp.argsort()

        #KNN算法核心部分

        for i in range(num):
            KNNmatrix.append(laberls[sortedDistIndices[i]])

        result = max(set(KNNmatrix),key=KNNmatrix.count)
        KNNmatrixNew.append(result)

    return KNNmatrixNew

group,laberls = TrainDateSet()
Tgroup = TestDateSet()
KNNmatrix = KNN(group=group,laberls=laberls,TestDateSetGroup=Tgroup,num=4)
print(KNNmatrix)









