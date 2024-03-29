import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score


# load admissions data
admissions_data = pd.read_csv("admissions_data.csv")
#print(admissions_data.head())
#admissions_data.describe()
#print(admissions_data.shape)

labels = admissions_data.iloc[:,-1]
#print(labels.describe())
features = admissions_data.iloc[:, 1:8]

# split our training and test set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state = 42)

# standardizing our data by scaling it
sc = StandardScaler()
features_train_scale = sc.fit_transform(features_train)
features_test_scale = sc.transform(features_test)

# check out the scaled data
#features_train_scale = pd.DataFrame(features_train_scale, columns = features_train.columns)
#features_test_scale = pd.DataFrame(features_test_scale, columns = features_test.columns)
#print(features_train_scale.describe())
#print(features_test_scale.describe())

# design a neural network model based on number of features
def design_model(features):
	model = Sequential()

	num_features = features.shape[1]
	input = tf.keras.Input(shape=(num_features))
	model.add(input)

	# this model has 2 hidden layers and 2 dropout layers
	# ReLU activation function is used at both hidden layers
	hidden_layer = layers.Dense(16, activation='relu')
	model.add(hidden_layer)
	model.add(layers.Dropout(0.1))

	hidden_layer_2 = layers.Dense(8, activation='relu')
	model.add(hidden_layer_2)
	model.add(layers.Dropout(0.2))

	# output layer with one node
	model.add(layers.Dense(1))

	# using an Adam optimizer with a learning rate of 0.005
	# using mean squared error as our loss function, and mean average error as our metric
	opt = keras.optimizers.Adam(learning_rate=0.005)
	model.compile(loss='mse', metrics=['mae'], optimizer=opt)

	return model


# apply the model to the scaled training data
model = design_model(features_train_scale)
#print(model.summary())

# apply early stopping for efficiency
# 1. monitor = 'val_loss' means we are monitoring the validation loss to decide when to stop the training
# 2. mode = 'min' means we seek minimal loss
# 3. patience = 20 means if the learning reaches a plateau, we continue for 20 more epochs in case the plateau leads to improved performance
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)

# fit the model with 100 epochs, batch size of 8, and validation split at 0.25
history = model.fit(features_train_scale, labels_train, epochs=100, batch_size=8, validation_split=0.25, callbacks=[es], verbose=1)

# evaluate the model
val_mse, val_mae = model.evaluate(features_test_scale, labels_test, verbose=0)

# check the mean average error (MAE) performance
print("MAE: ", val_mae)

# Evaluate R-squared score (coefficient of determination) to see how well the features in our regression model make predictions.
# An R-squared value near close to 1 suggests a well-fit regression model,
# while a value closer to 0 suggests that the regression model does not fit the data well.
labels_pred = model.predict(features_test_scale)
print("R-squared score: ", r2_score(labels_test, labels_pred))

# Plot the model's mean average error (MAE) per epoch for both training and validation data
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('Model MAE')
ax1.set_ylabel('MAE')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper left')

# Plot the model's loss per epoch for training and validation data
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper left')

plt.show()