import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the csv data as a DataFrame
df = pd.read_csv('telescope_data.csv', index_col=0)

# Remove null and na values
df.dropna()

# Print the DataFrame head
# print(df.head())

# Separate the numerical columns and the classes
classes = df['class']
data_matrix = df.drop(columns='class')

# Create and visualize a correlation matrix
correlation_matrix = data_matrix.corr()

ax = plt.axes()
sns.heatmap(correlation_matrix, cmap='Greens', ax=ax)
ax.set_title('Correlation Matrix')
plt.show()


# Perform eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
# print(f'Eigenvalues length: {eigenvalues.size}, Original Number of Features: {data_matrix.shape[1]}')

# Sort the eigenvalues and eigenvectors from greatest to smallest
indices = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[:, indices]
# print(eigenvalues.shape, eigenvectors.shape)

# Find the variance/information percentages for each eigenvalue.
information_proportions = eigenvalues / eigenvalues.sum()
information_percents = information_proportions * 100

# Plot the principal axes vs the information proportions for each principal axis
plt.figure()
plt.plot(information_percents, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Axes')
plt.ylabel('Percent of Information Explained')
plt.show()


# Find the cumulative variance/information percentages for each eigenvalue.
cumulative_information_percents = np.cumsum(information_percents)

# Plot the cumulative percentages array
plt.figure()
plt.plot(cumulative_information_percents, 'ro-', linewidth=2)

# Also plot a horizontal line indicating the 95% mark, and a vertical line for the sixth principal axis
plt.hlines(y=95, xmin=0, xmax=15)
plt.vlines(x=6, ymin=0, ymax=100)
plt.title('Cumulative Information percentages')
plt.xlabel('Principal Axes')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.show()
