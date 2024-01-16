import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
# transactions = pd.read_csv('transactions_modified.csv')
transactions = pd.read_csv('transactions.csv')

print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
print(transactions['isFraud'].sum())
print(transactions['isFraud'].sum() / transactions['isFraud'].count() * 100)

# Summary statistics on amount column
print(transactions['amount'].describe())

# Create isPayment field using Pandas' boolean indexing
transactions['isPayment'] = 0
transactions['isPayment'][transactions['type'].isin(['PAYMENT','DEBIT'])] = 1

# Create isMovement field
transactions['isMovement'] = 0
transactions['isMovement'][transactions['type'].isin(['CASH_OUT','TRANSFER'])] = 1

# Create accountDiff field
# Our theory is that destination accounts with a significantly different value could be suspect of fraud
transactions['accountDiff'] = abs(transactions['oldbalanceDest'] - transactions['oldbalanceOrg'])

# Create features and label variables
features = transactions[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = transactions['isFraud']

# Split dataset
f_train, f_test, l_train, l_test = train_test_split(features, label, test_size=0.3)

# Normalize the features variables
# Since sklearn‘s Logistic Regression implementation uses Regularization, we need to scale our feature data
scaler = StandardScaler()
f_train = scaler.fit_transform(f_train)
f_test = scaler.transform(f_test)

# Fit the model to the training data
regression = LogisticRegression()
regression.fit(f_train, l_train)

# Score the model on the training data
# The score returned is the percentage of correct classifications, or the accuracy
print(regression.score(f_train, l_train))

# Score the model on the test data
print(regression.score(f_test, l_test))

# Print the model coefficients
print(regression.coef_)

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
transaction4 = np.array([1000000.0, 0.0, 1.0, 1000000.0])

# Combine new transactions into a single array
sample_transactions = np.stack((transaction1, transaction2, transaction3, transaction4))

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
print(regression.predict(sample_transactions))

# Show probabilities on the new transactions
print(regression.predict_proba(sample_transactions))