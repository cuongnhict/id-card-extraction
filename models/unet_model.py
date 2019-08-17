from keras.layers import Input, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model


def unet(filters=8, width=512, height=512, n_channels=3, dropout=0.5, activation='relu', core_activation='sigmoid'):
  inputs = Input((width, height, n_channels))

  conv_1 = Conv2D(filters=filters, kernel_size=(3, 3), activation=activation, padding='same')(inputs)
  conv_1 = BatchNormalization()(conv_1)
  conv_1 = Conv2D(filters=filters, kernel_size=(3, 3), activation=activation, padding='same')(conv_1)
  conv_1 = BatchNormalization()(conv_1)
  pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
  pool_1 = Dropout(dropout * 0.5)(pool_1)

  conv_2 = Conv2D(filters=filters * 2, kernel_size=(3, 3), activation=activation, padding='same')(pool_1)
  conv_2 = BatchNormalization()(conv_2)
  conv_2 = Conv2D(filters=filters * 2, kernel_size=(3, 3), activation=activation, padding='same')(conv_2)
  conv_2 = BatchNormalization()(conv_2)
  pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
  pool_2 = Dropout(dropout)(pool_2)

  conv_3 = Conv2D(filters=filters * 4, kernel_size=(3, 3), activation=activation, padding='same')(pool_2)
  conv_3 = BatchNormalization()(conv_3)
  conv_3 = Conv2D(filters=filters * 4, kernel_size=(3, 3), activation=activation, padding='same')(conv_3)
  conv_3 = BatchNormalization()(conv_3)
  pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
  pool_3 = Dropout(dropout)(pool_3)

  conv_4 = Conv2D(filters=filters * 8, kernel_size=(3, 3), activation=activation, padding='same')(pool_3)
  conv_4 = BatchNormalization()(conv_4)
  conv_4 = Conv2D(filters=filters * 8, kernel_size=(3, 3), activation=activation, padding='same')(conv_4)
  conv_4 = BatchNormalization()(conv_4)
  pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)
  pool_4 = Dropout(dropout)(pool_4)

  conv_5 = Conv2D(filters=filters * 16, kernel_size=(3, 3), activation=activation, padding='same')(pool_4)
  conv_5 = BatchNormalization()(conv_5)
  conv_5 = Conv2D(filters=filters * 16, kernel_size=(3, 3), activation=activation, padding='same')(conv_5)
  conv_5 = BatchNormalization()(conv_5)

  up_6 = Conv2DTranspose(filters=filters * 8, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_5)
  up_6 = concatenate([up_6, conv_4])
  conv_6 = Dropout(dropout)(up_6)
  conv_6 = Conv2D(filters=filters * 8, kernel_size=(3, 3), activation=activation, padding='same')(conv_6)
  conv_6 = BatchNormalization()(conv_6)
  conv_6 = Conv2D(filters=filters * 8, kernel_size=(3, 3), activation=activation, padding='same')(conv_6)
  conv_6 = BatchNormalization()(conv_6)

  up_7 = Conv2DTranspose(filters=filters * 4, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_6)
  up_7 = concatenate([up_7, conv_3])
  conv_7 = Dropout(dropout)(up_7)
  conv_7 = Conv2D(filters=filters * 4, kernel_size=(3, 3), activation=activation, padding='same')(conv_7)
  conv_7 = BatchNormalization()(conv_7)
  conv_7 = Conv2D(filters=filters * 4, kernel_size=(3, 3), activation=activation, padding='same')(conv_7)
  conv_7 = BatchNormalization()(conv_7)

  up_8 = Conv2DTranspose(filters=filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_7)
  up_8 = concatenate([up_8, conv_2])
  conv_8 = Dropout(dropout)(up_8)
  conv_8 = Conv2D(filters=filters * 2, kernel_size=(3, 3), activation=activation, padding='same')(conv_8)
  conv_8 = BatchNormalization()(conv_8)
  conv_8 = Conv2D(filters=filters * 2, kernel_size=(3, 3), activation=activation, padding='same')(conv_8)
  conv_8 = BatchNormalization()(conv_8)

  up_9 = Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_8)
  up_9 = concatenate([up_9, conv_1])
  conv_9 = Dropout(dropout)(up_9)
  conv_9 = Conv2D(filters=filters, kernel_size=(3, 3), activation=activation, padding='same')(conv_9)
  conv_9 = BatchNormalization()(conv_9)
  conv_9 = Conv2D(filters=filters, kernel_size=(3, 3), activation=activation, padding='same')(conv_9)
  conv_9 = BatchNormalization()(conv_9)

  outputs = Conv2D(filters=1, kernel_size=(1, 1), activation=core_activation)(conv_9)

  model = Model(inputs=inputs, outputs=outputs)
  return model
