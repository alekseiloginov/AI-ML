import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

# Data loading and observing
dataset = pd.read_csv('life_expectancy.csv')
print(dataset.head())
print(dataset.describe())

# Drop the Country column. To create a predictive model, knowing from which country data comes can be confusing, and
# it is not a column we can generalize over. We want to learn a general pattern for all the countries,
# not only those dependent on specific countries.
dataset = dataset.drop(['Country'], axis = 1)

labels = dataset.iloc[:, -1]  # select all the rows (:), and access the last column (-1)
features = dataset.iloc[:, 0:-1]  # select all the rows (:), and access columns from 0 to the last column


# Data Preprocessing
# apply one-hot-encoding on all the categorical columns
features = pd.get_dummies(features)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

# standardize/normalize numerical features
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)


# Building the model
my_model = Sequential()

# Create an input layer to the network model with the number of nodes corresponding to the number of features in the dataset.
number_of_features = features.shape[1]
input = InputLayer(input_shape = (number_of_features,))
my_model.add(input)

# Add one fully-connected hidden layer with 64 hidden units and ReLU activation function
my_model.add(Dense(64, activation ="relu"))

# Add an output layer with one neuron as we need a single output for the regression prediction
my_model.add(Dense(1))

print(my_model.summary())


# Initializing the optimizer and compiling the model
optimizer = Adam(learning_rate = 0.01)

# Use Mean Squared Error (MSE) as loss function, and Mean Absolute Error (MAE) as additional metric for human-friendly analysis
my_model.compile(loss ='mse', metrics = ['mae'], optimizer = optimizer)


# Fit and evaluate the model
my_model.fit(features_train_scaled, labels_train, epochs = 40, batch_size = 1, verbose = 1)

res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose = 0)

print(res_mse, res_mae)
