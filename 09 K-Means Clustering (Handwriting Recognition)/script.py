import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Getting Started with the Digits Dataset:

digits = datasets.load_digits()
# print(digits)
# print(digits.DESCR)
# print(digits.data)
# print(digits.target)

# visualize the image at index 100:
plt.gray() # use grayscale colormap (instead of colorful default)
plt.matshow(digits.images[100]) # display an array as a matrix in a new figure window
plt.show()
# print(digits.target[100])

# take a look at 64 sample images

# Figure size (width, height)
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images
for i in range(64):

    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    # Label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

plt.show()

# K-Means Clustering:

model = KMeans(n_clusters=10, random_state=42) # random_state will ensure that every time you run your code, the model is built in the same way
model.fit(digits.data)

# Visualizing after K-Means:

# Let’s visualize all the centroids.
# Because data samples live in a 64-dimensional space, the centroids have values so they can be images.

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):

    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)

    # Display images of all the centroids (aka "cluster centers" as scikit-learn calls them).
    # The cluster centers should be a list with 64 values (0-16). And we are reshaping each of them into an 8x8 2D array.
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

    # Label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

plt.show()

# Testing the Model:

# predict labels for these four new handwritten digits (generated manually in test.html)
new_samples = np.array(
    [
        [0.00,3.13,6.63,6.79,3.74,0.08,0.00,0.00,2.59,7.62,5.57,5.26,7.62,1.45,0.00,0.00,3.66,6.86,0.08,1.60,7.62,1.52,0.00,0.00,0.31,0.92,0.00,3.89,7.55,0.84,0.00,0.00,0.00,0.00,0.92,7.32,4.88,0.00,0.00,0.00,0.00,0.99,6.71,7.40,1.75,1.52,1.53,0.31,0.00,3.59,7.62,7.62,7.62,7.62,7.62,3.51,0.00,0.61,2.29,2.29,2.29,1.83,1.53,0.30],
        [0.00,0.00,0.84,7.17,7.62,7.32,1.22,0.00,0.00,1.53,6.41,7.40,2.59,7.32,4.42,0.00,0.00,3.05,7.62,2.14,0.00,5.49,5.34,0.00,0.00,2.97,7.62,0.69,0.00,5.57,5.34,0.00,0.00,1.37,7.62,2.67,0.00,6.56,4.58,0.00,0.00,0.08,6.71,6.86,4.73,7.62,3.05,0.00,0.00,0.00,1.45,5.64,6.10,5.11,0.38,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
        [0.15,4.27,4.57,4.57,4.12,3.66,1.22,0.00,0.30,5.80,6.10,6.10,6.86,7.40,7.55,1.53,0.00,0.00,0.00,0.00,2.06,7.09,6.71,0.84,0.00,0.00,0.00,3.36,7.55,6.10,0.92,0.00,0.00,0.00,3.51,7.62,5.03,0.31,0.00,0.00,0.00,0.23,7.17,5.49,0.08,0.00,0.00,0.00,0.00,0.76,7.62,2.44,0.00,0.00,0.00,0.00,0.00,0.00,2.67,0.38,0.00,0.00,0.00,0.00],
        [0.00,0.84,4.50,4.57,4.57,4.96,2.59,0.00,0.00,2.44,7.62,6.25,6.10,6.10,2.59,0.00,0.00,5.19,7.17,0.31,0.00,0.00,0.00,0.00,0.00,7.63,7.02,5.26,3.20,0.30,0.00,0.00,0.00,4.73,5.34,5.95,7.63,4.27,0.00,0.00,0.00,1.53,0.61,1.14,7.32,4.27,0.00,0.00,0.00,6.79,6.48,7.17,7.17,0.84,0.00,0.00,0.00,3.05,6.10,4.73,1.07,0.00,0.00,0.00]
    ]
)

new_labels = model.predict(new_samples) # Returns index of the cluster each sample belongs to.
print(new_labels)

# Because this is a clustering algorithm, we don’t know which label is which.
# By looking at the cluster centers, let’s map out each of the labels with the digits we think it represents.

for i in range(len(new_labels)):
    if new_labels[i] == 0:
        print(2, end='')
    elif new_labels[i] == 1:
        print(0, end='')
    elif new_labels[i] == 2:
        print(1, end='')
    elif new_labels[i] == 3:
        print(8, end='')
    elif new_labels[i] == 4:
        print(7, end='')
    elif new_labels[i] == 5:
        print(6, end='')
    elif new_labels[i] == 6:
        print(3, end='')
    elif new_labels[i] == 7:
        print(5, end='')
    elif new_labels[i] == 8:
        print(9, end='')
    elif new_labels[i] == 9:
        print(4, end='')