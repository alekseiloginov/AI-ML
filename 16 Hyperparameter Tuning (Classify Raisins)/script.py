import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Explore the Dataset
raisins = pd.read_csv('Raisin_Dataset.csv')

# 1. Replace labels with 0/1 values
raisins['Class'] = raisins['Class'].replace({'Kecimen': 0, 'Besni': 1})
# raisins['Class'].replace({'Kecimen': 0, 'Besni': 1}, inplace=True)  # alternative
print(raisins.head())

# 2. Create predictor and target variables
X = raisins.drop(columns = 'Class')
# X = raisins.drop('Class', axis=1)  # alternative
y = raisins['Class']

# 3. Examine the dataset
print("Number of features:", X.shape[1])
print("Total number of samples:", len(X))
print("Samples belonging to class '1':", y.sum())

# 4. Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 19)


# Grid Search with Decision Tree Classifier

# 5. Create a Decision Tree model
tree = DecisionTreeClassifier()

# 6. Dictionary of parameters set up grid search to explore three values each for the following 2 hyperparameters:
# 'min_samples_split': The minimum number of samples to split at each node; let's explore the values 2,3 and 4.
# 'max_depth': The maximum tree depth; let's explore the values 3,5 and 7.
parameters = {'min_samples_split': [2,3,4], 'max_depth': [3,5,7]}

# 7. Create a GridSearchCV model
grid = GridSearchCV(tree, parameters)
# Fit the GridSearchCV model to the training data
grid.fit(X_train, y_train)

# 8. Print the model and hyperparameters obtained by GridSearchCV
print(grid.best_estimator_)
# Print best score
print(grid.best_score_)
# Print the accuracy of the final model on the test data
print(grid.score(X_test, y_test))

# 9. Print a table summarizing the results of GridSearchCV
df = pd.concat([
    pd.DataFrame(grid.cv_results_['params']),
    pd.DataFrame(grid.cv_results_['mean_test_score'], columns=['Score'])
], axis=1)
print(df)


# Random Search with Logistic Regression

# 10. The logistic regression model
lr = LogisticRegression(solver = 'liblinear', max_iter = 1000)

# 11. Define distributions to choose hyperparameters from
distributions = {'penalty': ['l1', 'l2'], 'C': uniform(loc=0, scale=100)}  # C - random value b/w 0 (loc) and 100 (loc+scale)

# 12. Create a RandomizedSearchCV model
clf = RandomizedSearchCV(lr, distributions, n_iter=8)  # perform random search 8 times

# Fit the random search model
clf.fit(X_train, y_train)

# 13. Print best estimator and best score
print(clf.best_estimator_)
print(clf.best_score_)

# Print a table summarizing the results of RandomSearchCV
df = pd.concat([
    pd.DataFrame(clf.cv_results_['params']),
    pd.DataFrame(clf.cv_results_['mean_test_score'], columns=['Accuracy'])
], axis=1)
print(df.sort_values('Accuracy', ascending = False))
