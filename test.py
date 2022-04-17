import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Actual data ranges from 1-255; This normalizes it down to the range of 0-1, which is easier to deal with
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.show()

# Input Layer (Flattening the data)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

# Hidden Layer:
# Defining the density of the layer and the activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Output Layer:
# Defining the output layer density (which is equal to the number of classifications) and the final
# activation function (which must give out a probability, thus softmax is chosen)
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Training Parameters
# Optimization of the training method, using conventional method like Stochastic Gradient-
# Descent; Calculation of the loss; Generation of Metrics to be tracked
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Finally, training the network
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)