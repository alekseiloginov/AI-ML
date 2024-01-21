import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# Load the data
health = pd.read_csv("dataR2.csv")
X = health.iloc[:,:-1]
y = health.iloc[:,-1]

# Logistic regression model
lr = LogisticRegression(max_iter=1000)

## Sequential forward selection
sfs = SFS(lr,
          k_features=3,
          forward=True,
          floating=False,
          scoring='accuracy',
          cv=0)
sfs.fit(X, y)

# Print the chosen feature names
print(sfs.subsets_[3]['feature_names'])
# Print the accuracy of the model after sequential forward selection
print(sfs.subsets_[3]['avg_score'])

# Plot the model accuracy
plot_sfs(sfs.get_metric_dict())
plt.show()

## Sequential backward selection
sbs = SFS(lr,
          k_features=3,
          forward=False,
          floating=False,
          scoring='accuracy',
          cv=0)
sbs.fit(X, y)

# Print the chosen feature names
print(sbs.subsets_[3]['feature_names'])
# Print the accuracy of the model after sequential backward selection
print(sbs.subsets_[3]['avg_score'])

# Plot the model accuracy
plot_sfs(sbs.get_metric_dict())
plt.show()

## Sequential forward floating selection
sffs = SFS(lr,
           k_features=3,
           forward=True,
           floating=True,
           scoring='accuracy',
           cv=0)
sffs.fit(X, y)

# Print a tuple with the names of the features chosen by sequential forward floating selection.
print(sffs.subsets_[3]['feature_names'])
# Print the accuracy
print(sffs.subsets_[3]['avg_score'])

## Sequential backward floating selection
sbfs = SFS(lr,
           k_features=3,
           forward=False,
           floating=True,
           scoring='accuracy',
           cv=0)
sbfs.fit(X, y)

# Print a tuple with the names of the features chosen by sequential backward floating selection.
print(sbfs.subsets_[3]['feature_names'])
# Print the accuracy
print(sbfs.subsets_[3]['avg_score'])

## Recursive Feature Elimination
# Create a list of feature names
features = list(X.columns)

# Standardize the data
X = StandardScaler().fit_transform(X)

# Logistic regression
lr = LogisticRegression(max_iter=1000)

# Recursive feature elimination
rfe = RFE(estimator=lr, n_features_to_select=3)
rfe.fit(X, y)

# List of features chosen by recursive feature elimination
rfe_features = [f for (f, support) in zip(features, rfe.support_) if support]
print(rfe_features)

# feature rankings after recursive feature elimination is done
print(rfe.ranking_)

# Print the accuracy of the model with features chosen by recursive feature elimination
print(rfe.score(X, y))
