import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('student_math.csv')

# Convert categorical features into new columns with 0/1 variables.
df = pd.get_dummies(df, drop_first = True)
print(df.info())
print(df.columns, df.shape)
print(df.head())


# Implementing Regularization with Linear Regression

# We set our predictor and outcome variables and perform a train-test split:
y = df['G3']
X = df.drop(columns = ['G3'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit a Lasso regularized regression model:
lasso = Lasso(alpha = 0.05)
lasso.fit(X_train,y_train)

# Look at the Mean Squared Error (MSE):
pred_train = lasso.predict(X_train)
pred_test = lasso.predict(X_test)
training_mse = mean_squared_error(y_train, pred_train)
test_mse = mean_squared_error(y_test, pred_test)
print('Training Error:',  training_mse)
print('Test Error:', test_mse)


# Tuning the Regularization Hyperparameter

# One way to figure out the sweet spot in this bias-variance tradeoff is to try multiple values of alpha.
# Iterating over an array of alpha values and plotting the resulting training and test errors against the corresponding alpha's
# would get us a figure that we can inspect to get a sense of the optimal alpha value.


# Automate the L1 Hyperparameter Search with GridsearchCV

# Implement Lasso regularization with GridsearchCV to do multiple train-test splits to cover the entire sample
tuned_parameters = [{'alpha': np.logspace(-6, 0, 100)}]  # 100 alpha values between 0.000001 and 1.0
folds = 5
model = GridSearchCV(estimator = Lasso(), param_grid = tuned_parameters, scoring = 'neg_mean_squared_error', cv = folds, return_train_score = True)
model.fit(X, y)

# cv_results_ gives us the details of every model fit corresponding to a particular alpha value and the train-test split of each fold.
# We’re specifically going to look at the mean train and test scores across the 5 train-test splits:
test_scores = model.cv_results_['mean_test_score']
train_scores = model.cv_results_['mean_train_score']

# Get the alpha that is optimal to our scoring strategy:
print(model.best_params_, model.best_score_)


# L2 Hyperparameter Search with GridsearchCV

# Create an array of 100 alpha values between 0.01 and 10000, and put into hash used as parameter grid
tuned_parameters = [{'alpha': np.logspace(-2, 4, 100)}]

# Perform GridSearchCV with Ridge regularization on the data
model = GridSearchCV(estimator = Ridge(), param_grid = tuned_parameters, scoring = 'neg_mean_squared_error', cv = 5, return_train_score = True)
model.fit(X, y)

# Print the tuned alpha and the best test score corresponding to it
print(model.best_params_, model.best_score_)
