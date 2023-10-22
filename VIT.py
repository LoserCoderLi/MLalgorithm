import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image,ImageOps
import os

model = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

train_dir = "Z:\\archive\\Alzheimer_s Dataset\\train"
test_dir = "Z:\\archive\\Alzheimer_s Dataset\\test"

Y_train = []
X_train = []
label = 0

for i in os.listdir(train_dir):
    print(i)
    for j in os.listdir(train_dir+"/"+i):
        # 做不掉就丢的方法
        try:
        # PIL
            img = Image.open(train_dir+"/"+i+"/"+j)
            img = img.resize((150,150))
            # PID
            img = ImageOps.colorize(img,black="black",white="white")
            img = model(images=img)
            # sklearn
            img = img["pixel_values"]
            img = np.array(img)

            X_train.append(img.flatten()/255)
            Y_train.append(label)
        except Exception as e:
            print("error is ",e)
    label += 1

Y_test = []
X_test = []
label = 0

for i in os.listdir(test_dir):
    print(i)
    for j in os.listdir(test_dir+"/"+i):
        # 做不掉就丢的方法
        try:
            # PIL
            img = Image.open(test_dir+"/"+i+"/"+j)
            img = img.resize((150,150))
            # PID
            img = ImageOps.colorize(img,black="black",white="white")
            img = model(images=img)
            # sklearn
            img = img["pixel_values"]
            img = np.array(img)

            X_test.append(img.flatten()/255)
            Y_test.append(label)
        except Exception as e:
            print("error is ",e)
    label += 1

from sklearn import linear_model

model = linear_model.LogisticRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)


# Classifier
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, pred)
print(cm)
print("accuracy is", (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/sum(sum(cm)))

from sklearn import ensemble
model = ensemble.RandomForestClassifier()

model.fit(X_train,Y_train)
pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
print("accuracy is", (cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/sum(sum(cm)))

