import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle


NAME = "GoogLeNet-Plant-Model-batch-32-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# Loading in the data
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
y = np.array(y)
# Normalizing data
X = X/255.0

def inception(x,
              filters_1x1,
              filters_3x3_reduce,
              filters_3x3,
              filters_5x5_reduce,
              filters_5x5,
              filters_pool):

    path1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding="same", activation="relu") (x)

    path2 = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding="same", activation="relu") (x)
    path2 = tf.keras.layers.Conv2D(filters_3x3, (1, 1), padding="same", activation="relu") (path2)

    path3 = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding="same", activation="relu") (x)
    path3 = tf.keras.layers.Conv2D(filters_5x5, (1, 1), padding="same", activation="relu") (path3)

    path4 = tf.keras.layers.MaxPool2D( (3, 3), strides=(1, 1), padding="same") (x)
    path4 = tf.keras.layers.Conv2D(filters_pool, (1, 1), padding="same", activation="relu") (path4)

    return tf.concat([path1, path2, path3, path4], axis=3)


inp = tf.keras.layers.Input(shape=(256, 256, 1))

input_tensor = tf.keras.layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=X.shape[1:]) (inp)

x = tf.keras.layers.Conv2D(64, 7, strides=2, padding="same", activation="relu") (input_tensor)
x = tf.keras.layers.MaxPooling2D(3, strides=2) (x)

x = tf.keras.layers.Conv2D(64, 1, strides=1, padding="same", activation="relu")(x)
x = tf.keras.layers.Conv2D(192, 3, strides=1, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D(3, strides=2) (x)

x = inception(x, 64, 96, 128, 16, 32, 32)

x = inception(x, 128, 128, 192, 32, 96, 64)

x = tf.keras.layers.MaxPooling2D(3, strides=2) (x)

x = inception(x, 192, 96, 208, 16, 48, 64)

aux1 = tf.keras.layers.AveragePooling2D((5, 5), strides=3) (x)
aux1 = tf.keras.layers.Conv2D(128, 1, padding="same", activation="relu") (aux1)
aux1 = tf.keras.layers.Flatten() (aux1)
aux1 = tf.keras.layers.Dense(1024, activation="relu") (aux1)
aux1 = tf.keras.layers.Dropout(0.7) (aux1)
aux1 = tf.keras.layers.Dense(10, activation="softmax") (aux1)

x = inception(x, 160, 112, 224, 24, 64, 64)

x = inception(x, 128, 128, 256, 24, 64, 64)

x = inception(x, 112, 144, 288, 32, 64, 64)

aux2 = tf.keras.layers.AveragePooling2D((5, 5), strides=3) (x)
aux2 = tf.keras.layers.Conv2D(128, 1, padding="same", activation="relu") (aux2)
aux2 = tf.keras.layers.Flatten() (aux2)
aux2 = tf.keras.layers.Dense(1024, activation="relu") (aux2)
aux2 = tf.keras.layers.Dropout(0.7) (aux2)
aux2 = tf.keras.layers.Dense(10, activation="softmax") (aux2)

x = inception(x, 256, 160, 320, 32, 128, 128)

x = tf.keras.layers.MaxPooling2D(3, strides=2) (x)

x = inception(x, 256, 160, 320, 32, 128, 128)

x = inception(x, 384, 192, 384, 48, 128, 128)

x = tf.keras.layers.GlobalAveragePooling2D() (x)

x = tf.keras.layers.Dropout(0.4) (x)

out = tf.keras.layers.Dense(10, activation="softmax") (x)

model = Model(inputs=inp, outputs=[out, aux1, aux2])

model.compile(optimizer="adam", loss=[losses.sparse_categorical_crossentropy, losses.sparse_categorical_crossentropy, losses.sparse_categorical_crossentropy], loss_weights=[1, 0.3, 0.3], metrics=["accuracy"])

history = model.fit(X, y, batch_size=32, epochs=3, validation_split=0.2, callbacks=[tensorboard])