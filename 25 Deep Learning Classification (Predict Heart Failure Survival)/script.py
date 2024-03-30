import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

# Loading the data
data = pd.read_csv('heart_failure.csv')
print(data.info())

# check how imbalanced is the number of classes in the data
print('Classes and number of values in the dataset:', Counter(data['death_event']))

X = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]
y = data["death_event"]


# Data preprocessing
X = pd.get_dummies(X)  # one hot encoding of categorical data

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# scale numeric features in the dataset
ct = ColumnTransformer([("numeric", Normalizer(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)


# Prepare labels for classification
# encode categorical classes into integers
le = LabelEncoder()
Y_train = le.fit_transform(Y_train.astype(str))
Y_test = le.transform(Y_test.astype(str))

# one hot encoding of integers into binary vectors, so we can use categorical_crossentropy loss on them
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)


# Design the model
model = Sequential()

model.add(InputLayer(input_shape=(X_train.shape[1],)))  # size of the features
model.add(Dense(12, activation='relu'))  # hidden layer
model.add(Dense(2, activation='softmax'))  # output layer, use softmax for classification, size 2 to match possible classes

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train and evaluate the model
model.fit(X_train, Y_train, epochs = 100, batch_size = 16, verbose=1)

loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print("Loss:", loss, "Accuracy:", acc)


# Generating a classification report
y_estimate = model.predict(X_test, verbose=0)
y_estimate = np.argmax(y_estimate, axis=1)  # select index of the true class for each binary vector label encoding in y_estimate

y_true = np.argmax(Y_test, axis=1)

# check additional metrics, such as F1-score
print(classification_report(y_true, y_estimate))
