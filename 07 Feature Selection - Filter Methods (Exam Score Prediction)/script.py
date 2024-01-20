import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from functools import partial

df = pd.DataFrame(data={
    'edu_goal': ['bachelors', 'bachelors', 'bachelors', 'masters', 'masters', 'masters', 'masters', 'phd', 'phd', 'phd'],
    'hours_study': [1, 2, 3, 3, 3, 4, 3, 4, 5, 5],
    'hours_TV': [4, 3, 4, 3, 2, 3, 2, 2, 1, 1],
    'hours_sleep': [10, 10, 8, 8, 6, 6, 8, 8, 10, 10],
    'height_cm': [155, 151, 160, 160, 156, 150, 164, 151, 158, 152],
    'grade_level': [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    'exam_score': [71, 72, 78, 79, 85, 86, 92, 93, 99, 100]
})
print(df)

# 10 x 6 features matrix
X = df.drop(columns=['exam_score'])
print(X)

# 10 x 1 target vector
y = df['exam_score']
print(y)

## Variance threshold
# remove non-numeric features
X_num = X.drop(columns=['edu_goal'])
print(X_num)

selector = VarianceThreshold(threshold=0)  # 0 is default
print(selector.fit_transform(X_num))

# Specify `indices=True` to get indices of selected features
print(selector.get_support(indices=True))

# Use indices to get the corresponding column names of selected features
num_cols = list(X_num.columns[selector.get_support(indices=True)])
print(num_cols)

# Subset `X_num` to retain only selected features
X_num = X_num[num_cols]
print(X_num)

# Finally, obtain our entire features DataFrame, including the categorical column edu_goal
X = X[['edu_goal'] + num_cols]
print(X)

## Pearson’s correlation
# Correlation between features
corr_matrix = X_num.corr(method='pearson')  # 'pearson' is default

sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r')
plt.show()

# Let's define high correlation as having a coefficient of greater than 0.7 or less than -0.7.
# We can loop through the correlation matrix to identify the highly correlated variables.

# Loop over bottom diagonal of correlation matrix
for i in range(len(corr_matrix.columns)):
    for j in range(i):

        # Print variables with high correlation
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            print(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])

# Correlation between feature and target
X_y = X_num.copy()
X_y['exam_score'] = y
print(X_y)

# Generate the correlation matrix
corr_matrix = X_y.corr()

# Isolate the column corresponding to `exam_score`
corr_target = corr_matrix[['exam_score']].drop(labels=['exam_score'])

sns.heatmap(corr_target, annot=True, fmt='.3', cmap='RdBu_r')
plt.show()

# Let’s remove hours_TV as the redundant feature:
X = X.drop(columns=['hours_TV'])
print(X)

# Find the F-statistic for a model with each predictor on its own
print(f_regression(X_num, y))

## Mutual information
# Encode edu_goal column using the LabelEncoder
le = LabelEncoder()

# Create copy of `X` for encoded version
X_enc = X.copy()
X_enc['edu_goal'] = le.fit_transform(X['edu_goal'])
print(X_enc)

# Compute mutual information between each feature and exam_score
print(mutual_info_regression(X_enc, y, random_state=68))

# To properly calculate the mutual information, we need to tell mutual_info_regression() which features are discrete
# by providing their index positions using the discrete_features parameter.
print(mutual_info_regression(X_enc, y, discrete_features=[0], random_state=68))

# Pick out the top k features with the highest ranked scores
# Use partial() function to specify additional arguments for our scoring function
score_func = partial(mutual_info_regression, discrete_features=[0], random_state=68)

# Select top 3 features with the most mutual information
selection = SelectKBest(score_func=score_func, k=3)
print(selection.fit_transform(X_enc, y))

# Subset our original features DataFrame with the indices of the selected features
X = X[X.columns[selection.get_support(indices=True)]]
print(X)
