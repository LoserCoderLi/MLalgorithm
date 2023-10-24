import pandas as pd

class DataFiller:
    def __init__(self, data):
        self.data = data

    def zero_fill(self):
        self.data = self.data.fillna(value=0)
        return self.data

    def ward_fill(self, back):
        if back:
            self.data = self.data.fillna(method='bfill')
        else:
            self.data = self.data.fillna(method='ffill')
        return self.data

    def mean_fill(self):
        for x in self.data:
            self.data[x] = self.data[x].fillna(value=self.data[x].mean())
        return self.data


train_data = pd.read_csv("H:\\data\\train_data .csv")
test_data = pd.read_csv("H:\\data\\test_data.csv")

from sklearn.preprocessing import StandardScaler

# 创建一个StandardScaler对象
scaler = StandardScaler()

# 使用训练数据拟合scaler
scaler.fit(train_data)
scaler.fit(test_data)

# 使用scaler转换训练数据和测试数据
X_train_scaled = scaler.transform(train_data)
X_test_scaled = scaler.transform(test_data)


X_train = train_data.iloc[:, 2:-1] # 取第二列到倒数第二列
Y_train = train_data.iloc[:, -1] # 取最后一列
print(train_data)

Fillmethod_train = DataFiller(X_train)

X_train_0 = Fillmethod_train.zero_fill() # 0填充
X_train_1 = Fillmethod_train.mean_fill() # 均值填充
X_train_2 = Fillmethod_train.ward_fill(False) # 使用前向填充
X_train_3 = Fillmethod_train.ward_fill(True) # 使用后向填充

X_test = test_data.iloc[:, 2:-1]
Y_test = test_data.iloc[:, -1]

Fillmethod_test = DataFiller(X_test)

X_test_0 = Fillmethod_test.zero_fill() # 0填充
X_test_1 = Fillmethod_test.mean_fill() # 均值填充
X_test_2 = Fillmethod_test.ward_fill(False) # 使用前向填充
X_test_3 = Fillmethod_test.ward_fill(True) # 使用后向填充

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression



def Model(X_train, Y_train, X_test, Y_test, model):
    if model == 'tree':
        Tree = tree.DecisionTreeClassifier()
        Tree = Tree.fit(X_train, Y_train)

        TY_pred = Tree.predict(X_test)
        TY_pred_train = Tree.predict(X_train)
        accuracy = accuracy_score(Y_test, TY_pred)
        print('Tree\'s accuracy is ', accuracy)

        # 打印分类报告
        print('Train:Classification Report for Decision Tree:')
        print(classification_report(Y_train, TY_pred_train))
        print('Test:Classification Report for Decision Tree:')
        print(classification_report(Y_test, TY_pred))

        return accuracy

    elif model == 'knn':
        best_score = 0.0
        best_k = 0
        for k in range(1, 11):
            Knn = KNeighborsClassifier(n_neighbors=k)
            Knn.fit(X_train, Y_train)

            KY_pred = Knn.predict(X_test)
            KY_pred_train = Knn.predict(X_train)
            accuracy = accuracy_score(Y_test, KY_pred)
            if accuracy > best_score:
                best_k = k
                best_score = accuracy
        print('best_k = ', best_k)
        print('Knn\'s the best accuracy is ', best_score)

        # 打印分类报告
        print('Train:Classification Report for KNN with best k:')
        print(classification_report(Y_train, KY_pred_train))
        print('Test:Classification Report for KNN with best k =', best_k)
        print(classification_report(Y_test, KY_pred))

        return best_score

    elif model == 'nn':
        nn = MLPClassifier(max_iter=100000)  # you might need to increase max_iter for convergence
        nn.fit(X_train, Y_train)

        NN_pred = nn.predict(X_test)
        NN_pred_train = nn.predict(X_train)
        accuracy = accuracy_score(Y_test, NN_pred)
        print('Neural Network\'s accuracy is ', accuracy)

        print('Train:Classification Report for Neural Network:')
        print(classification_report(Y_train, NN_pred_train))
        print('Test:Classification Report for Neural Network:')
        print(classification_report(Y_test, NN_pred))

        return accuracy
    elif model == 'lr':  # logistic regression
        lr = LogisticRegression(multi_class='multinomial',
                                max_iter=1000)  # you might need to increase max_iter for convergence
        lr.fit(X_train, Y_train)

        LR_pred = lr.predict(X_test)
        LR_pred_train = lr.predict(X_train)
        accuracy = accuracy_score(Y_test, LR_pred)
        print('Logistic Regression\'s accuracy is ', accuracy)

        print('Train:Classification Report for Logistic Regression:')
        print(classification_report(Y_train, LR_pred_train))
        print('Test:Classification Report for Logistic Regression:')
        print(classification_report(Y_test, LR_pred))

        return accuracy


# Knn = KNeighborsClassifier()
# Knn.fit(X_train_0, Y_train)
#
# KY_pred = Knn.predict(X_test_0)
# accuracy = accuracy_score(Y_test, KY_pred)
# print('Knn\'s accuracy is ', accuracy)


Taccuracy_0 = Model(X_train_0, Y_train, X_test_0, Y_test, model='tree')
Taccuracy_1 = Model(X_train_1, Y_train, X_test_1, Y_test, model='tree')
Taccuracy_2 = Model(X_train_2, Y_train, X_test_2, Y_test, model='tree')
Taccuracy_3 = Model(X_train_3, Y_train, X_test_3, Y_test, model='tree')

# Kaccuracy_0, k_0 = Model(X_train_0, Y_train, X_test_0, Y_test, model='knn')
# Kaccuracy_1, k_1 = Model(X_train_1, Y_train, X_test_1, Y_test, model='knn')
# Kaccuracy_2, k_2 = Model(X_train_2, Y_train, X_test_2, Y_test, model='knn')
# Kaccuracy_3, k_3 = Model(X_train_3, Y_train, X_test_3, Y_test, model='knn')

# Naccuracy_0 = Model(X_train_0, Y_train, X_test_0, Y_test, model='nn')
# Naccuracy_1 = Model(X_train_1, Y_train, X_test_1, Y_test, model='nn')
# Naccuracy_2 = Model(X_train_2, Y_train, X_test_2, Y_test, model='nn')
# Naccuracy_3 = Model(X_train_3, Y_train, X_test_3, Y_test, model='nn')

Laccuracy_0 = Model(X_train_0, Y_train, X_test_0, Y_test, model='lr')
Laccuracy_1 = Model(X_train_1, Y_train, X_test_1, Y_test, model='lr')
Laccuracy_2 = Model(X_train_2, Y_train, X_test_2, Y_test, model='lr')
Laccuracy_3 = Model(X_train_3, Y_train, X_test_3, Y_test, model='lr')

