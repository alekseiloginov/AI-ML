import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Read the csv data as a DataFrame
df = pd.read_csv('telescope_data.csv', index_col=0)

# Remove null and na values
df.dropna()

# Separate the numerical columns and the classes
classes = df['class']
data_matrix = df.drop(columns='class')

# Standardize the data matrix
mean = data_matrix.mean(axis=0)
std = data_matrix.std(axis=0)
data_matrix_standardized = (data_matrix - mean) / std


# Perform PCA by fitting and transforming the data matrix.
pca = PCA()

# Fit the standardized data and calculate the principal components
principal_components = pca.fit_transform(data_matrix_standardized)
# print(f'Number of features in the principal components: {principal_components.shape[1]}')
# print(f'Number of features in the data matrix: {data_matrix.shape[1]}')


# Calculate the eigenvalues from the singular values and extract the eigenvectors.
singular_values = pca.singular_values_
eigenvalues = singular_values ** 2

# Eigenvectors are in the property `.components_` as row vectors. To turn them into column vectors, transpose them using the NumPy method `.T`.
eigenvectors = pca.components_.T


# Extract the variance ratios, which are equivalent to the eigenvalue proportions.
principal_axes_variance_ratios = pca.explained_variance_ratio_
principal_axes_variance_percents = principal_axes_variance_ratios * 100


# Perform PCA once again but with 2 components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_matrix_standardized)
# print(f'Number of Principal Components Features: {principal_components.shape[1]}')
# print(f'Number of Original Data Features: {data_matrix_standardized.shape[1]}')


# Plot the principal components and have its class as its hue to see if clustering of any kind has occurred.
principal_components_data = pd.DataFrame({
    'PC1': principal_components[:, 0],
    'PC2': principal_components[:, 1],
    'class': classes,
})

# Use the seaborn function .lmplot for a scatter plot with a hue based on the label ‘class’.
# sns.lmplot(x='PC1', y='PC2', data=principal_components_data, hue='class', fit_reg=False)
# plt.show()


# We will use the one-hot-encoded (0/1) classes as the y
y = classes.astype('category').cat.codes


# Fit the transformed features using 2 principal components onto the classifier and generate a score.
pca_1 = PCA(n_components=2)

# Use the principal components as X and split the data into 33% testing and the rest training
X = pca_1.fit_transform(data_matrix_standardized)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create a Linear Support Vector Classifier
svc_1 = LinearSVC(random_state=0, tol=1e-5)
svc_1.fit(X_train, y_train)

# Generate a score for the testing data
score_1 = svc_1.score(X_test, y_test)
print(f'Score for model with 2 PCA features: {score_1}')


# Now, fit the classifier with the first 2 features of the original data matrix and generate a score.
# Notice the large difference in scores!

# Select two features from the original data
first_two_original_features = [0, 1]
X_original = data_matrix_standardized.iloc[:, first_two_original_features]

# Split the data intro 33% testing and the rest training
X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.33, random_state=42)

# Create a Linear Support Vector Classifier
svc_2 = LinearSVC(random_state=0)
svc_2.fit(X_train, y_train)

# Generate a score for the testing data
score_2 = svc_2.score(X_test, y_test)
print(f'Score for model with 2 original features: {score_2}')
