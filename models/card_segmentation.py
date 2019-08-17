import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from model.unet_model import unet

warnings.filterwarnings('ignore')

X = np.load('../dataset/final_image.npy')
y = np.load('../dataset/final_mask.npy')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
X_train = X_train / 255.0
y_train = y_train / 255.0
X_val = X_val / 255.0
y_val = y_val / 255.0

callbacks = [
  EarlyStopping(min_delta=1e-4, patience=10, verbose=1),
  TensorBoard(log_dir='../logs/', batch_size=8, write_graph=True),
  ModelCheckpoint('card_segmentation_60epochs.h5', monitor='val_loss', verbose=1, save_best_only=True)
]

model = unet()
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=60, callbacks=callbacks, validation_data=(X_val, y_val))
