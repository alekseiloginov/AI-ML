import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Construct an ImageDataGenerator object:
DIRECTORY = "Covid19-dataset/train"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256,256)
BATCH_SIZE = 32
training_data_generator = ImageDataGenerator(rescale=1.0/255,
                                             zoom_range=0.1, # Randomly increase or decrease the size of the image by up to 10%
                                             rotation_range=25, # Randomly rotate the image between -25,25 degrees
                                             width_shift_range=0.05, # Randomly shift the image along its width by up to +/- 5%
                                             height_shift_range=0.05) # Randomly shift the image along its height by up to +/- 5%
validation_data_generator = ImageDataGenerator()

print("Loading training data...")
training_iterator = training_data_generator.flow_from_directory(DIRECTORY,class_mode='categorical',color_mode='grayscale',batch_size=BATCH_SIZE)#, subset='training')
training_iterator.next()

print("\nLoading validation data...")
validation_iterator = validation_data_generator.flow_from_directory(DIRECTORY,class_mode='categorical', color_mode='grayscale',batch_size=BATCH_SIZE)#, subset='validation')
# Print its attributes:
# print(training_data_generator.__dict__)

print("\nBuilding model...")
def design_model(training_data):
    # sequential model
    model = Sequential()

    # add input layer with grayscale image shape
    model.add(tf.keras.Input(shape=(256, 256, 1)))

    # two convolutional hidden layers with relu functions, maxpooling layers and dropout layers as well
    model.add(layers.Conv2D(5, 5, strides=3, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(3, 3, strides=1, activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(layers.Dropout(0.2))

    # experimenting with extra layers
    #model.add(tf.keras.layers.Conv2D(3, 3, strides=1, activation="relu"))
    #model.add(tf.keras.layers.Conv2D(1, 1, strides=1, activation="relu"))
    #model.add(tf.keras.layers.Dropout(0.1))

    model.add(layers.Flatten())

    # output layer with softmax activation function
    model.add(layers.Dense(3,activation="softmax"))

    # compile model with Adam optimizer
    # loss function is categorical cross-entropy
    # metrics are categorical accuracy and AUC
    print("\nCompiling model...")
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss = tf.keras.losses.CategoricalCrossentropy(),
                  metrics = [tf.keras.metrics.CategoricalAccuracy(),
                             tf.keras.metrics.AUC()])

    # summarize model
    model.summary()

    return model


# use model function
model = design_model(training_iterator)

# early stopping implementation
es = EarlyStopping(monitor='val_auc', mode='min', verbose=1, patience=20)

# fit the model with 10 epochs and early stopping
print("\nTraining model...")
history = model.fit(
    training_iterator,
    steps_per_epoch = training_iterator.samples/BATCH_SIZE,
    epochs=5,
    validation_data = validation_iterator,
    validation_steps = validation_iterator.samples/BATCH_SIZE,
    callbacks = [es]
)

# plotting categorical accuracy (cross-entropy loss) for train and validation data over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting AUC for train and validation data over epochs over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['auc'])
ax2.plot(history.history['val_auc'])
ax2.set_title('model auc')
ax2.set_xlabel('epoch')
ax2.set_ylabel('auc')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()

# evaluate false positives and false negatives
# implementing a classification report
test_steps_per_epoch = math.ceil(validation_iterator.samples / validation_iterator.batch_size)
predictions = model.predict(validation_iterator, steps=test_steps_per_epoch)
predicted_classes = numpy.argmax(predictions, axis=1)

true_classes = validation_iterator.classes

class_labels = list(validation_iterator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# implementing a confusion matrix
cm = confusion_matrix(true_classes,predicted_classes)
print(cm)
