import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# load compressed galaxy data using NumPy
data = np.load('galaxydata.npz')
input_data = data["data"]
labels = data["labels"]
# print(input_data)
# print(labels)
print(input_data.shape)
print(labels.shape)

# divide the data into training and validation data
# Set `stratify=labels` to ensure that ratios of galaxies in the testing data are the same as in the original dataset.
x_train, x_valid, y_train, y_valid = train_test_split(input_data, labels, test_size=0.20, stratify=labels, shuffle=True, random_state=222)

# preprocess the input - normalize the pixel values into a range between 0 and 1
data_generator = ImageDataGenerator(rescale=1./255)

# create input data iterators, each time reading a batch of 5 images
BATCH_SIZE = 5
training_iterator = data_generator.flow(x_train, y_train, batch_size=BATCH_SIZE)
validation_iterator = data_generator.flow(x_valid, y_valid, batch_size=BATCH_SIZE)

# build a CNN model
model = tf.keras.Sequential()

# add input layer, having the shape of the data: 128 x 128 images with 3 color channels (RGB)
model.add(tf.keras.Input(shape=(128, 128, 3)))

# add two convolutional layers with eight 3x3 filters, interspersed with max pooling layers having 2x2 windows
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# followed by a flatten layer and a dense layer
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation="relu"))

# add output layer: classify result using softmax into 4 classes (“Normal”, ”Ringed”, ”Merger”, ”Other”).
model.add(tf.keras.layers.Dense(4, activation="softmax"))

# compile the model with an optimizer, loss, and metrics for accuracy & AUC
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics = [tf.keras.metrics.CategoricalAccuracy(),
               tf.keras.metrics.AUC()]
)

# check model's shape and parameters
model.summary()

# train the model
model.fit(
    training_iterator,
    steps_per_epoch = len(x_train) / BATCH_SIZE,
    epochs = 8,
    validation_data = validation_iterator,
    validation_steps = len(x_valid) / BATCH_SIZE
)

# Visualize how the convolutional neural network processes images
from visualize import visualize_activations
# This function loads in sample data, uses the model to make predictions, and then saves the feature maps from each convolutional layer.
# These feature maps showcase the activations of each filter as they are convolved across the input.
# `visualize_results` takes the Keras model and the validation iterator and does the following:
# 1. It loads in a sample batch of data using the validation iterator.
# 2. It uses model.predict() to generate predictions for the first sample images.
# 3. Next, it compares those predictions with the true labels and prints the result.
# 4. It then shows the image and the feature maps for each convolutional layer using matplotlib.
visualize_activations(model, validation_iterator)
