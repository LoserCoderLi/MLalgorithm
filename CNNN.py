from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from joblib import dump
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense,Input, Dropout
from sklearn.model_selection import train_test_split

from keras.applications.resnet50 import ResNet50
from keras import optimizers

def draw_(x):
    for number in range(64):
        plt.subplot(8, 8, number+1)
        plt.tight_layout()
        # print('sssssssssss')
        # print(x.train_data[number])
        plt.imshow(x/255, interpolation='none', cmap='gray')
        # plt.title("Title: {}".format(x.train_labels[number]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


train_dir = "F:\\modleeeee\\archive1\\chest_xray\\train"
train_data = ImageDataGenerator().flow_from_directory(train_dir,(150,150),batch_size=5,shuffle=False)

test_dir = "F:\\modleeeee\\archive1\\chest_xray\\test"
test_data = ImageDataGenerator().flow_from_directory(test_dir,(150,150),batch_size=5,shuffle=False)## (路径，大小，数量，洗牌)
# print(dir(test_data))
# for x in test_data:
#     # print(x)
#     for y in x:
#         print("yyyyy")
#         print(y)
#         print(y.shape)## 数量，大小，通道
#         for z in y:
#             print("zzzzzzzzzzzzzzzzzzz")
#             print(z)
#             print(z.shape)
#             draw_(z)
#             break
#         break
#     break
# print(test_data)
print(train_data)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Flatten())
model.summary()

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))
# model.add(print())
model.add(Dense(2, activation="softmax"))
# model.add(Flatten())
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4,activation='softmax'))
#
# model = Model(input=ResNet50.input, output=model(ResNet50.output))
# model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),metrics=['accuracy'])
#
# model.summary()
#

## metrics:网络评价指标
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# for z in train_data:
#     print(z)
#     for x in z:
#         print(x)
#     break+

print(type(train_data))

h = model.fit(train_data, batch_size=5, epochs=3)
# print(dir(h))

plt.plot(h.history['accuracy'])
#plt.plot(h.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(h.history['loss'])
#plt.plot(h.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


pred = model.predict(test_data)
pred = np.argmax(pred,axis=1)


cm = confusion_matrix(test_data.classes,pred)
print(cm)
print("accuracy is", (cm[0,0]+cm[1,1])/sum(sum(cm)))

dump(model, 'F:\\modleeeee\\CNNNNN.joblib')
print("ok")