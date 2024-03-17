import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('water_potability.csv')
df = df.dropna()  # LogisticRegression does not accept missing values
print(df.columns, df.shape)

# Make a train-test split of our data to handle the training process
X = df.drop(['Potability'], axis=1)
y = df['Potability']
rand_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=rand_state)

# To assemble our ensemble, make a dictionary of base estimators.
level_0_estimators = dict()
level_0_estimators["logreg"] = LogisticRegression(random_state=rand_state)
level_0_estimators["forest"] = RandomForestClassifier(random_state=rand_state)

# Prepare to add new features to our training dataset (as columns).
level_0_columns = [f"{name}_prediction" for name in level_0_estimators.keys()]

# Our final estimator will be a Random Forest.
level_1_estimator = RandomForestClassifier(random_state=rand_state)

# Use Stratified K-Folds cross-validator as a splitting strategy used to train the final estimator.
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=rand_state)

# The kfold is then given to the StackingClassifier.
# Base estimators are fitted on full X, and final estimator - using cross-validated predictions of base estimators.
stacking_clf = StackingClassifier(estimators=list(level_0_estimators.items()),
                                  final_estimator=level_1_estimator,
                                  passthrough=True,
                                  cv=kfold,
                                  stack_method="predict_proba")

# fit_transform trains our base and final estimators, makes cross-validated predictions,
# and augment training set with predictions from each estimator.
X_augmented = stacking_clf.fit_transform(X_train, y_train)

# Let's see how the resulting training dataset looks
df = pd.DataFrame(X_augmented, columns=level_0_columns + list(X_train.columns))
print(df.head())
print(y_train.head())

# Make predictions and compare how Stacking classifier performed with how a linear model / decision tree model perform
y_val_pred = stacking_clf.predict(X_test)
stacking_accuracy = accuracy_score(y_test, y_val_pred)

logistic_regression = LogisticRegression(random_state=rand_state).fit(X_train, y_train)
lr_accuracy = accuracy_score(y_test, logistic_regression.predict(X_test))

decision_tree = RandomForestClassifier(random_state=rand_state).fit(X_train, y_train)
dt_accuracy =  accuracy_score(y_test, decision_tree.predict(X_test))

print(f'Stacking accuracy: {stacking_accuracy:.4f}')
print(f'Logistic Regression accuracy: {lr_accuracy:.4f}')
print(f'Decision Tree accuracy: {dt_accuracy:.4f}')
