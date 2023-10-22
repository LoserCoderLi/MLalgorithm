import os, cv2

import numpy as np

from tensorflow.keras.applications import resnet50,ResNet50
from tensorflow.keras.preprocessing import image


from PIL import Image

import os
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense,Input, Dropout

import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras import optimizers
from sklearn.metrics import confusion_matrix


# model = Sequential()
model=ResNet50(weights='imagenet', include_top=False, input_shape=(150,150,3))
# model.add(Dense(4, activation="softmax"))

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
train_dir = "Z:\\archive1\\Alzheimer_s Dataset\\train"
test_dir = "Z:\\archive1\\Alzheimer_s Dataset\\test"

# train_data = ImageDataGenerator().flow_from_directory(train_dir,(150,150),batch_size=1,shuffle=False)
#
# for x in train_data:
#     print(x.shape)


# print(train_data.shape)
# model.fit(train_data,epochs=30, batch_size=6)


Y_train = []
X_train = []
label = 0

for i in os.listdir(train_dir):
    print(i)

    for j in os.listdir(train_dir + "/" + i):
        # img = Image.open(train_dir + "/" + i + "/" + j)
        # img = img.convert("RGB")
        # img = img.resize((150, 150))
        # data = np.asarray(img)

        # print(train_dir + "/" + i)
        # print(train_dir + "/" + i + "/" + j)
        img = cv2.imread(train_dir + "/" + i + "/" + j)
        img = cv2.resize(img, (150, 150))
        # print(j)
        # print(img)
        img = image.img_to_array(img)
        # print(img.shape)
        # print(img)
        img = resnet50.preprocess_input(np.expand_dims(img.copy(), axis=0))

        img = model.predict(img)
        #
        X_train.append(img.flatten() / 255)


        # img = cv2.imread(test_dir+"/"+i+"/"+j)
        # img = cv2.resize(img, (150,150))
        # # print(j)
        # # print(img)
        # img = image.img_to_array(img)
        # # print(img)
        # img = resnet50.preprocess_input(np.expand_dims(img.copy(), axis=0))
        # print(data.shape)
        #
        # X_train.append(img)
        Y_train.append(label)

    label += 1

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = X_train.astype('float32')
X_train = X_train/255.0



Y_test = []
X_test = []
label = 0

for i in os.listdir(test_dir):
    print(i)

    for j in os.listdir(test_dir + "/" + i):
        # img = Image.open(test_dir + "/" + i + "/" + j)
        # img = img.convert("RGB")
        # img = img.resize((150, 150))
        # data = np.asarray(img)

        img = cv2.imread(test_dir + "/" + i + "/" + j)
        img = cv2.resize(img, (150, 150))
        # print(j)
        # print(img)
        img = image.img_to_array(img)
        # print(img)
        img = resnet50.preprocess_input(np.expand_dims(img.copy(), axis=0))
        # 旧的
        img = model.predict(img)
        #
        X_test.append(img.flatten() / 255)

        # print(data.shape)
        # X_test.append(img)
        Y_test.append(label)

    label += 1
#
# X_test = np.array(X_test)
# Y_test = np.array(Y_test)
# X_test = X_test.astype('float32')
# X_test = X_test/255.0
# # print(Y_train)

from keras import optimizers
from matplotlib import pyplot as plt

# print(type(X_train))
# print(type(Y_train))

X_train = np.array(X_train)
Y_train = np.array(Y_train)
# X_train = X_train.astype('float32')
# Y_train = Y_train.astype('float32')

model = Sequential()
# model.add(ResNet50(weights='imagenet', include_top=False, input_shape=(150,150,3)))
# model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.2))
# model.add(print(img.shape))
model.add(Dense(4, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

result = model.fit(X_train,Y_train,epochs=33,batch_size=16)


# input_tensor = Input(shape=(150,150,3))
# ResNet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
#
# top_model = Sequential()
# top_model.add(ResNet50.predict())
# top_model.add(Flatten()/255)
# top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(4, activation='softmax'))
#
# top_model = Model(inputs=ResNet50.input,outputs=top_model(ResNet50.output))
# top_model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),metrics=['accuracy'])
#
# top_model.summary()
# result = top_model.fit(X_train,Y_train,epochs=30,batch_size=16)

plt.plot(result.history['accuracy'])
#plt.plot(h.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(result.history['loss'])
#plt.plot(h.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()



# model = linear_model.LogisticRegression()
# for x in X_train:
#     print(x.shape)
# model.fit(X_train, Y_train)
pred = model.predict(np.array(X_test))
pred = np.argmax(pred,axis=1)


# Classifier
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, pred)
print(cm)
print("accuracy is", (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/sum(sum(cm)))

# pred = model.predict(Y_test)
# pred = np.argmax(pred,axis=1)
#
#
# cm = confusion_matrix(Y_test,pred)
# print(cm)
# print("accuracy is", (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/sum(sum(cm)))
# for i in os.listdir(train_dir):
#     print(i)
#
#     for j in os.listdir(train_dir+"/"+i):
#         print(train_dir+"/"+i)
#         print(train_dir+"/"+i+"/"+j)
#         img = cv2.imread(train_dir+"/"+i+"/"+j)
#         print(img.shape)
#         img = cv2.resize(img,(150,150))
#         print(img.shape)
#
#
#         img = image.img_to_array(img)

#
#         img = resnet50.preprocess_input(np.expand_dims(img.copy(), axis=0))
#         img = model.predict(img)
#         print(img.shape)
#
#         X_train.append(img.flatten()/255)
#         Y_train.append(label)
#     label += 1
#
#
# Y_test = []
# X_test = []
# label = 0
# for i in os.listdir(test_dir):
#     print(i)
#
#     for j in os.listdir(test_dir+"/"+i):
#         img = cv2.imread(test_dir+"/"+i+"/"+j)
#         img = cv2.resize(img, (150,150))
#         # print(j)
#         # print(img)
#         img = image.img_to_array(img)
#         # print(img)
#         img = resnet50.preprocess_input(np.expand_dims(img.copy(), axis=0))
#         img = model.predict(img)
#
#         X_test.append(img.flatten()/255)
#         Y_test.append(label)
#     label += 1
#
#
from sklearn import linear_model

model = linear_model.LogisticRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
#
#
# Classifier
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, pred)
print(cm)
print("accuracy is", (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/sum(sum(cm)))


# # Clustering
# from yellowbrick.cluster import KElbowVisualizer
# v = KElbowVisualizer(model,k=(2,20))
# v.fit(np.array(X_train))
# v.show()
# print("ok")

from sklearn.neighbors import KNeighborsClassifier
best_score = 0.0
best_k = -1
for k in range(1,11):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, Y_train)
    score = knn_clf.score(X_test, Y_test)
    if score > best_score:
        best_k = k
        best_score = score
print("best_k = ", best_k)
print("best_score = ", best_score)


