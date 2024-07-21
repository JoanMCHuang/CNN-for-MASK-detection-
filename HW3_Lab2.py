#pip install opencv-python keras

import os
import cv2
import numpy as np
from keras.utils import np_utils

# 設定圖像大小和路徑
img_rows, img_cols = 112, 112
data_path = 'data'
classes = ['without_mask', 'with_mask']

# 讀取圖像資料集
def get_data():
    data = []
    labels = []
    for cls in classes:
        path = os.path.join(data_path, cls)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                resized_img = cv2.resize(image, (img_rows, img_cols))
                data.append(resized_img)
                labels.append(cls)
            except Exception as e:
                print(e)
    return data, labels

# 將圖像資料集分為訓練集和測試集
def get_train_test_data():
    data, labels = get_data()
    labels = np.array(labels)
    data = np.array(data)
    num_classes = len(classes)
    labels = np_utils.to_categorical(labels, num_classes)
    # 將資料隨機分為訓練集和測試集
    indices = np.random.permutation(data.shape[0])
    data = data[indices]
    labels = labels[indices]
    num_train_samples = int(data.shape[0] * 0.8)
    x_train, y_train = data[:num_train_samples], labels[:num_train_samples]
    x_test, y_test = data[num_train_samples:], labels[num_train_samples:]
    return x_train, y_train, x_test, y_test

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# 建立CNN模型
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_rows, img_cols, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu')) 
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model
    batch_size = 128
    epochs = 10
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))

    # Evaluate model on test data
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score

    # 模型存檔
    model.save('model.h5')

    # 模型載入
    model = tf.keras.models.load_model('model.h5')