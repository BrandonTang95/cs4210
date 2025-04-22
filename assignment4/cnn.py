# -------------------------------------------------------------------------
# AUTHOR: Brandon Tang
# FILENAME: cnn.py
# SPECIFICATION: The program will build and train a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify 32x32 images of handwritten digits (0 through 9).
# FOR: CS 4210 - Assignment #4
# TIME SPENT: 1 hour
# -------------------------------------------------------------------------

# Importing Python libraries
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models  # type: ignore

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Add at the top of your code


# Function to load dataset
def load_digit_images_from_folder(folder_path, image_size=(32, 32)):
    X = []
    y = []
    for filename in os.listdir(folder_path):
        # Getting the label of the image (it's the first number in the filename)
        label = int(filename.split("_")[0])

        # Converting images to grayscale and resizing them to 32x32
        img = (
            Image.open(os.path.join(folder_path, filename))
            .convert("L")
            .resize(image_size)
        )

        # Adding the converted image to the feature matrix and label to the class vector
        X.append(np.array(img))
        y.append(label)
    return np.array(X), np.array(y)


# Set your own paths here (relative to your project folder)
train_path = os.path.join(
    "digit_dataset", "train"
)  # Changed from "images" to "digit_dataset"
test_path = os.path.join(
    "digit_dataset", "test"
)  # Changed from "images" to "digit_dataset"

# Loading the raw images
X_train, Y_train = load_digit_images_from_folder(train_path)
X_test, Y_test = load_digit_images_from_folder(test_path)

# Normalizing the data: convert pixel values from range [0, 255] to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshaping the input images to include the channel dimension: (num_images, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)

# Building a CNN model
model = models.Sequential(
    [
        # Convolutional layer with 32 filters of size 3x3, relu activation, and input shape 32x32x1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 1)),
        # Max pooling layer with pool size 2x2
        layers.MaxPooling2D((2, 2)),
        # Flatten layer to convert the feature maps into a 1D vector
        layers.Flatten(),
        # Dense (fully connected) layer with 64 neurons and relu activation
        layers.Dense(64, activation="relu"),
        # Output layer with 10 neurons (digits 0-9) and softmax activation
        layers.Dense(10, activation="softmax"),
    ]
)

# Compiling the model
model.compile(
    optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Fitting the model
model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test))

# Evaluating the model on the test set
loss, acc = model.evaluate(X_test, Y_test)

# Printing the test accuracy
print(f"Test accuracy: {acc:.4f}")
