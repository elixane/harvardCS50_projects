import cv2 as cv
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # Create empty lists for images and labels
    images = []
    labels = []

    # # Loop through each category folder (0 to NUM_CATEGORIES-1)
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))

        # Ensure the directory exists
        if not os.path.isdir(category_dir):
            continue

        # Loop through each image file in the category directory
        for img_file in os.listdir(category_dir):
            img_path = os.path.join(category_dir, img_file)

            # Read the image file and check if it was read correctly
            img = cv.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            # Resize image to correct measurements
            img = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Convert image to numpy array
            img_array = np.array(img)

            # Add the image array and corresponding label to their lists
            images.append(img_array)
            labels.append(category)

    # Return the tuple (images, labels)
    return (images, labels)

    

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        # First convolutional layer with 32 filters and a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # Batch normalization to normalize activations
        tf.keras.layers.BatchNormalization(),

        # Max-pooling layer
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer with 64 filters
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Third convolutional layer with 128 filters
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the 3D outputs to 1D
        tf.keras.layers.Flatten(),

        # Fully connected hidden layer with 512 units
        tf.keras.layers.Dense(512, activation="relu"),
        
        # Dropout for regularization
        tf.keras.layers.Dropout(0.5),

        # Fully connected hidden layer with 256 units
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output layer with NUM_CATEGORIES units and softmax activation
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
