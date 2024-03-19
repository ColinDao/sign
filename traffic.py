import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Number of dataset iterations through the model
EPOCHS = 10

# Image size parameters
IMG_WIDTH = 30
IMG_HEIGHT = 30

# Number of different images
NUM_CATEGORIES = 43

# Test data is 40% of dataset
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
        np.array(images), np.array(labels), test_size = TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs = EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose = 2)

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

    # List of the sign types
    dirs = os.listdir(data_dir)

    # Lists to store the images and their corresponding label
    images = []
    labels = []

    # Loop through the sign types
    for filename in dirs:

        # Get the path from the directory to the sign type
        dir_to_file = os.path.join(data_dir, filename)

        # List of the pictures of the given sign type
        pictures = os.listdir(dir_to_file)

        # Loop through the pictures
        for image in pictures:

            # Image from the `directory -> sign type -> image` path
            temp_image = cv2.imread(os.path.join(dir_to_file, image))

            # Resize the image to desired parameters
            resized = cv2.resize(temp_image, (IMG_WIDTH, IMG_HEIGHT))

            # Add the resized image and its type to the datasets
            images.append(resized)
            labels.append(filename)

    # Return the images and labels datasets
    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    # Image shape
    image_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

    # Convolution kernel size
    conv_kernel = (3, 3)

    # Pooling kernel size
    pool_kernel = (2, 2)

    """
    Types of layers:

    Convolution: Weighted kernels used to extract features from nearby pixels.
                 Each filter for the layer has its own kernel

        Ex.: Edges, blobs, textures, etc.

    Max Pooling: Weighted kernels used to extract the most prominent feature 
                 from nearby pixels. Less sensitive to small shifts, more focused
                 on capturing the prominent features

        Ex.: Get brightest color (possibly indicating an outline) from an area
             of the image

    Flatten: Convert multi-dimensional features into a one-dimensional vector

        Ex.: Converting a 2d array to a 1d array

    Dense: Weighted hidden neurons to model non-linear data. Learn the complex
           relations between data

        Ex.: Detect if it's a picture of a hotdog or a burger

    Dropout: Drop internal neurons after the current layer with probability `p`

        Ex.: Drop neurons A and M with probability 50%
    """

    # Create a neural network model
    model = tf.keras.models.Sequential([

        # Convolutional layer: learn 32 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            32, conv_kernel, activation = "relu", input_shape = image_shape
        ),

        # Max-pooling layer: 2x2 kernel
        tf.keras.layers.MaxPooling2D(pool_size = pool_kernel),

        # Convolutional layer: learn 64 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            64, conv_kernel, activation = "relu", input_shape = image_shape
        ),

        # Max-pooling layer: 2x2 kernel
        tf.keras.layers.MaxPooling2D(pool_size = pool_kernel),

        # Convolutional layer: learn 256 filters using a 3x3 kernel
        tf.keras.layers.Conv2D(
            256, conv_kernel, activation = "relu", input_shape = image_shape
        ),

        # Max-pooling layer: 2x2 kernel
        tf.keras.layers.MaxPooling2D(pool_size = pool_kernel),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Hidden layer: 256 neurons
        tf.keras.layers.Dense(256, activation = "relu"),

        # Dropout layer: 30% chance of internal neurons not being considered (dropped)
        tf.keras.layers.Dropout(0.3),

        # Hidden layer: 256 neurons
        tf.keras.layers.Dense(256, activation = "relu"),

        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")

    ])
    

    model.compile(

        # Algorithm used to minimize errors
        optimizer = "Adamax",

        # Loss function
        loss = "categorical_crossentropy",

        # Metrics used to evaluate the model
        metrics = ["accuracy"],

        # Automatically determine the number of shards for parallel evaluation
        pss_evaluation_shards = "auto"
    )

    return model

if __name__ == "__main__":
    main()