# %%
import os
import cv2
import numpy as np
import pandas as pd

from tensorflow.python.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.python.keras import Model, Sequential, regularizers
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.nadam import Nadam
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from keras.losses import SparseCategoricalCrossentropy, sparse_categorical_crossentropy
from keras.losses import BinaryCrossentropy

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#%%
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#%%
import tensorflow as tf
tf.test.is_gpu_available()

# %%
# 이미지 학습 데이터 불러오기
members = ['jhope', 'jimin', 'jin', 'rm', 'suga', 'v', 'jk']
X = []

for member in members:
    files = os.listdir(f"./cutting_faces/{member}/")
    set = []

    for i, path in enumerate(files):
        img = cv2.imread(f"./cutting_faces/{member}/"+path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        set.append(gray)

    X.append(np.array(set).reshape(-1,180,180,1))
# %%
for i in range(len(X)):
    print(f"{members[i]} shape :", X[i].shape)

# %%
# y data set 만들기
jhope_y = np.full((152,1), 0)
jimin_y = np.full((220,1), 1)
jin_y = np.full((196,1), 2)
rm_y = np.full((212,1), 3)
suga_y = np.full((156,1), 4)
v_y = np.full((124,1), 5)
jk_y = np.full((283,1), 6)

# %%
X_set = np.concatenate(X)
y_set = np.concatenate([jhope_y, jimin_y, jin_y, rm_y, suga_y, v_y, jk_y])

# %%
print(X_set.shape)
print(y_set.shape)

#%%
X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.25)
print("X_train shape :", X_train.shape)
print("y_train shape :", y_train.shape)
print("X_test shape :", X_test.shape)
print("y_test shape :", y_test.shape)

# %%
HIDDEN_UNITS = 64
# BATCH_SIZE = 8
# opti = Adam(learning_rate=0.01)
opti = Nadam(learning_rate=0.003)
lossFunc = SparseCategoricalCrossentropy()
# lossFunc = sparse_categorical_crossentropy()

NORM = regularizers.l2(0.1)

model = Sequential()
model.add(Conv2D(HIDDEN_UNITS, kernel_size=(7,7), activation='relu', input_shape=X_set.shape[1:], kernel_regularizer=NORM))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=(7,7), activation='relu', kernel_regularizer=NORM))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
# model.add(Dense(16, activation='elu'))
model.add(Dense(7, activation='softmax'))

model.summary()
model.compile(optimizer=opti, loss=lossFunc, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=70, verbose=1)

# %%
# 모델 test
# prediction은 확률값을 반환하는데 이 때, 가장 높은 확률값이 예측 클래스가 됨.
preds = model.predict(X_test)

predictions = []
for pred in preds:
    predictions.append(np.argmax(pred))
    
pred = np.array(predictions).reshape(-1,1)

acc = np.round(accuracy_score(y_test, pred), 3)
print(acc)

# %%
# df = pd.DataFrame(pred, columns=['pred'])
# df['y_test'] = y_test

model.evaluate(X_test, y_test, verbose=1)

# %%
