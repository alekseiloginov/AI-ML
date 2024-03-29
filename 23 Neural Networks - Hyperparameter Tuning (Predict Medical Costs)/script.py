import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from sklearn.dummy import DummyRegressor
from scipy.stats import randint as sp_randint

tf.random.set_seed(42) # for reproducibility of result we always use the same seed for random number generator
dataset = pd.read_csv("insurance.csv") # read the dataset


# Prepare data and model
def design_model(X, learning_rate, simple=True):  # our function to design the model
    model = Sequential(name="my_sequential_model")

    input = tf.keras.Input(shape=(X.shape[1],))  # input layer will have as many nodes as the number of features
    model.add(input)

    # adding hidden layer(s)
    if simple:
        model.add(layers.Dense(64, activation='relu'))
    else:
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))

    model.add(layers.Dense(1)) # output layer

    opt = Adam(learning_rate = learning_rate) # setting the learning rate of Adam to the one specified in the function parameter
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)

    return model

features = dataset.iloc[:,0:6] # choose first 7 columns as features
labels = dataset.iloc[:,-1] # choose the final column for prediction

# one hot encoding for categorical variables
# Force newly created columns to be represented as integers instead of booleans.
# This ensures that they are compatible with TensorFlow's numerical tensor requirements.
features = pd.get_dummies(features, dtype=int)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)

# standardize numerical columns
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train) # gives numpy arrays
features_test = ct.transform(features_test) # gives numpy arrays


# Manual tuning: learning rate
def fit_model(f_train, l_train, learning_rate, num_epochs, batch_size):
    # build the model
    model = design_model(f_train, learning_rate)
    # train the model on the training data
    history = model.fit(f_train, l_train, epochs = num_epochs, batch_size = batch_size, verbose = 0, validation_split = 0.2)
    # plot learning curves
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('learn rate = ' + str(learning_rate))
    plt.legend(loc="upper right")

# make a list of learning rates to try out
learning_rates = [1, 0.01, 0.001, 1E-7]
# fixed number of epochs
num_epochs = 100
# fixed number of batches
batch_size = 10

for i in range(len(learning_rates)):
    plot_no = 420 + (i+1)
    plt.subplot(plot_no)
    fit_model(features_train, labels_train, learning_rates[i], num_epochs, batch_size)

plt.tight_layout()
plt.show()


# Manual tuning: batch size
def fit_model(f_train, l_train, learning_rate, num_epochs, batch_size, ax):
    model = design_model(features_train, learning_rate)
    # train the model on the training data
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size = batch_size, verbose=0, validation_split = 0.3)
    # plot learning curves
    ax.plot(history.history['mae'], label='train')
    ax.plot(history.history['val_mae'], label='validation')
    ax.set_title('batch = ' + str(batch_size), fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax.set_xlabel('# epochs')
    ax.set_ylabel('mae')
    ax.legend()

# fixed learning rate
# learning_rate = 0.01
learning_rate = 0.1
print("Learning rate fixed to:", learning_rate)

# fixed number of epochs
num_epochs = 100

# we choose a number of batch sizes to try out
# batches = [1, 2, 16]
batches = [4, 32, 64]

# plotting code
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', sharey='row', gridspec_kw={'hspace': 0.7, 'wspace': 0.4}) # preparing axes for plotting
axes = [ax1, ax2, ax3]

# iterate through all the batch values
for i in range(len(batches)):
    fit_model(features_train, labels_train, learning_rate, num_epochs, batches[i], axes[i])

plt.show()


# Manual tuning: epochs and early stopping
def fit_model(f_train, l_train, learning_rate, num_epochs):
    # build the model
    # we can increase the number of hidden neurons in order to introduce some overfitting
    model = design_model(features_train, learning_rate, simple=False)
    # train the model on the training data
    # 1. without early stopping:
    # history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size= 16, verbose=0, validation_split = 0.2)
    # 2. with early stopping:
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20)
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size= 16, verbose=0, validation_split = 0.2, callbacks = [es])

    return history

learning_rate = 0.1
num_epochs = 500
history = fit_model(features_train, labels_train, learning_rate, num_epochs)

# plotting loss
fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 1, 'wspace': 0.5})
(ax1, ax2) = axs
ax1.plot(history.history['loss'], label='train')
ax1.plot(history.history['val_loss'], label='validation')
ax1.set_title('lrate=' + str(learning_rate))
ax1.legend(loc="upper right")
ax1.set_xlabel("# of epochs")
ax1.set_ylabel("loss (mse)")

# plotting MAE
ax2.plot(history.history['mae'], label='train')
ax2.plot(history.history['val_mae'], label='validation')
ax2.set_title('lrate=' + str(learning_rate))
ax2.legend(loc="upper right")
ax2.set_xlabel("# of epochs")
ax2.set_ylabel("MAE")

print("Final training MAE:", history.history['mae'][-1])
print("Final validation MAE:", history.history['val_mae'][-1])

plt.show()


# Manual tuning: changing the model
def no_hidden_layer_model(X, learning_rate):
    model = Sequential(name="my_sequential_model")
    input = tf.keras.Input(shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

def one_hidden_layer_model(X, learning_rate):
    model = Sequential(name="my_sequential_model")
    input = tf.keras.Input(shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(8, activation='relu'))
    # model.add(layers.Dense(64, activation='relu'))  # to compare when early stopping happens
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

def fit_model(model, f_train, l_train, learning_rate, num_epochs):
    # train the model on the training data
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20)
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size= 2, verbose=0, validation_split = 0.2, callbacks = [es])
    return history

def plot(history):
    # plot learning curves
    fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 1, 'wspace': 0.8})
    (ax1, ax2) = axs

    # plot loss
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='validation')
    ax1.set_title('lrate=' + str(learning_rate))
    ax1.legend(loc="upper right")
    ax1.set_xlabel("# of epochs")
    ax1.set_ylabel("loss (mse)")

    # plot MAE
    ax2.plot(history.history['mae'], label='train')
    ax2.plot(history.history['val_mae'], label='validation')
    ax2.set_title('lrate=' + str(learning_rate))
    ax2.legend(loc="upper right")
    ax2.set_xlabel("# of epochs")
    ax2.set_ylabel("MAE")

    print("Final training MAE:", history.history['mae'][-1])
    print("Final validation MAE:", history.history['val_mae'][-1])

learning_rate = 0.1
num_epochs = 200

# fit the more simple model
print("Results of a model with no hidden layers:")
history1 = fit_model(no_hidden_layer_model(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)
plot(history1)
plt.show()

#fit the more complex model
print("Results of a model with one hidden layer:")
history2 = fit_model(one_hidden_layer_model(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)
plot(history2)
plt.show()


# Automated tuning: grid and random search
#------------- GRID SEARCH --------------
def do_grid_search():
    batch_size = [6, 64]
    epochs = [10, 50]
    # To use GridSearchCV from scikit-learn for regression we need to wrap our neural network model into a KerasRegressor
    model = KerasRegressor(build_fn=design_model)
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator = model, param_grid=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False), return_train_score=True)
    grid_result = grid.fit(features_train, labels_train, verbose = 0)
    print(grid_result)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    print("Traininig")
    means = grid_result.cv_results_['mean_train_score']
    stds = grid_result.cv_results_['std_train_score']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

#------------- RANDOMIZED SEARCH --------------
def do_randomized_search():
    param_grid = {'batch_size': sp_randint(2, 16), 'nb_epoch': sp_randint(10, 100)}
    model = KerasRegressor(build_fn=design_model)
    grid = RandomizedSearchCV(estimator = model, param_distributions=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False), n_iter = 12)
    grid_result = grid.fit(features_train, labels_train, verbose = 0)
    print(grid_result)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

print("-------------- GRID SEARCH --------------------")
do_grid_search()
print("-------------- RANDOMIZED SEARCH --------------------")
do_randomized_search()


# Regularization: dropout
def design_model_dropout(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = tf.keras.Input(shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

def design_model_no_dropout(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = layers.InputLayer(input_shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

learning_rate = 0.001
num_epochs = 200

# train the model without dropout
history1 = fit_model(design_model_no_dropout(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)
plot(history1)
plt.show()

# train the model with dropout
history2 = fit_model(design_model_dropout(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)
plot(history2)
plt.show()


# Baselines
# Let's see if the model's performance is reasonable, meaning it does better than a baseline.
# For our regression task we can use averages or medians of the class distribution known as central tendency measures.
# Scikit-learn provides DummyRegressor, which serves as a baseline regression algorithm.
# Let's try mean (average) and median as our central tendency measures.
dummy_regr = DummyRegressor(strategy="mean")
# dummy_regr = DummyRegressor(strategy="median")
dummy_regr.fit(features_train, labels_train)
labels_pred = dummy_regr.predict(features_test)
MAE_baseline = mean_absolute_error(labels_test, labels_pred)
print('MAE baseline: $', MAE_baseline)
