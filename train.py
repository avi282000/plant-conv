import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle


NAME = "Plant-Model-64x2-batch-32-bias-True-layer-1-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# Loading in the data
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))
y = np.array(y)
# Normalizing data
X = X/255.0
print(X)
model = Sequential()

# Adding the Convolutional layer (3, 3) is the window size
# and the shape is the size of the image matrix (i.e. 256x256x1)
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:], use_bias=True))
# Activation Function
model.add(Activation("relu"))
# Pooling Layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Adding it again (Making it a 64x2 Conv Net)
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Final Fully Connected Layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard])