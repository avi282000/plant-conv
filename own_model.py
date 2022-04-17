import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = r"YOUR_DATASET_PATH_HERE"
CATEGORIES = ["Pepper Bell", "Tomato"]

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # For defining the paths for each plant type
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img),
                                   cv2.IMREAD_GRAYSCALE)  # Choosing each picture and converting it to grayscale
            # print(img_array)
            # print(img_array.shape)
            # plt.imshow(img_array, cmap="gray")
            # plt.show()
            training_data.append([img_array, class_num])


create_training_data()

print(len(training_data))

# Shuffle the data (pictures)
random.shuffle(training_data)
for sample in training_data:
    print(sample[1])

#
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 256, 256, 1)
print(type(X))
# Reshape each image into 256x256 and into grayscale (-1 = ALL THE FEATURES
# WILL BE COLLECTED; 256 = THE SIZE OF THE IMAGE; 1 = GRAYSCALE (TURN IT TO 3 for BRG))


# Dumping all the parameters out (So as to not resize all the pics again and do other memory-intensive tasks)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)

# Use this to load these parameters again into another model
# pickle_in = open("X.pickle","rb")
# X = pickle.load(pickle_in)

