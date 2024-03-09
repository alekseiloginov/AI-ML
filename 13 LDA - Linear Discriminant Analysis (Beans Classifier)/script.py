import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load data
beans = pd.read_csv('beans.csv')
X = beans.drop('Class', axis=1)
y = beans['Class']

# Create an LDA model
lda = LinearDiscriminantAnalysis()

# Fit lda to X and y and create a subspace X_new
X_new = lda.fit_transform(X, y)

# Create a logistic regression model
lr = LogisticRegression()

# Fit lr to X_new and y
lr.fit(X_new, y)

# Model accuracy of the logistic regression model
lr_acc = lr.score(X_new, y)
print(lr_acc)


# LDA as a classifier

# We can also use LDA itself as a classifier.
# In this case, scikit-learn will project the data onto a subspace and then find a decision boundary that is perpendicular to that subspace.
# Put simply, it will do LDA and use the result to find a linear decision boundary.

# When using LDA as a classifier, it’s not necessary to transform X into X_new.
# You can simply use the fit() method to fit lda to X and y.

# Load data
beans = pd.read_csv('beans.csv')
X = beans.drop('Class', axis=1)
y = beans['Class']

# Create LDA model
lda = LinearDiscriminantAnalysis(n_components=1)

# Fit the data and create a subspace X_new
lda.fit(X, y)

# Print LDA classifier accuracy
print(lda.score(X, y))
