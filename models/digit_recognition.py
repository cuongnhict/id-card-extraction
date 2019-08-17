import warnings
import glob
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

warnings.filterwarnings('ignore')

images = []
labels = []
for file_path in glob.glob('../dataset/numbers/*'):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (29, 44))
    image = np.expand_dims(image, axis=2)
    images.append(image)

    file_name = os.path.basename(file_path)
    label = file_name[0]
    labels.append(int(label))

X = np.array(images)
y = np.array(labels).reshape((-1, 1))
y = OneHotEncoder().fit_transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(44, 29, 1)))
model.add(Conv2D(32, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

callbacks = [
    EarlyStopping(min_delta=1e-4, patience=10, verbose=1),
    ModelCheckpoint('digit_100epochs.h5', monitor='val_loss', verbose=1, save_best_only=True)
]

model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=callbacks, batch_size=32, epochs=100, verbose=1)
