import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from itertools import product

# Creating and visualizing AND Data
data = [[0,0], [0,1], [1,0], [1,1]]
labels = [0, 0, 0, 1]

# Let's plot these four points on a graph
plt.scatter([point [0] for point in data],
            [point [1] for point in data],
            c=labels)
# plt.show()


# Building AND Perceptron
classifier = Perceptron(max_iter=40, random_state = 42)
classifier.fit(data, labels)

# Let's see if the algorithm learned AND
print(classifier.score(data, labels))


# Visualizing decision boundary of AND Perceptron
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))   # returns distance b/w points and decision boundary

# We use this function for a grid of points to make a heat map that reveals the decision boundary

# Create a list of the points we want to input to decision_function()
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

# Find every possible combination of those x and y values
point_grid = list(product(x_values, y_values))

distances = classifier.decision_function(point_grid)

# Right now distances stores positive and negative values.
# We only care about how far away a point is from the boundary, so we can drop the sign.
abs_distances = [abs(pt) for pt in distances]

# Right now abs_distances is a list of 10000 numbers, and pcolormesh() needs a two dimensional list.
# We need to turn abs_distances into a 100 by 100 two dimensional array.
distances_matrix = np.reshape(abs_distances, (100, 100))

# Draw the heat map
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heatmap)  # put a legend on the heat map
# plt.show()


# Creating and visualizing XOR Data
data = [[0,0], [0,1], [1,0], [1,1]]
labels = [0, 1, 1, 0]

# Let's plot these four points on a graph
plt.scatter([point [0] for point in data],
            [point [1] for point in data],
            c=labels)
# plt.show()


# Building XOR Perceptron
classifier = Perceptron(max_iter=40, random_state = 42)
classifier.fit(data, labels)

# Let's see if the algorithm learned XOR
print(classifier.score(data, labels))


# Visualizing decision boundary of XOR Perceptron

# Create a list of the points we want to input to decision_function()
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

# Find every possible combination of those x and y values
point_grid = list(product(x_values, y_values))

distances = classifier.decision_function(point_grid)

# Right now distances stores positive and negative values.
# We only care about how far away a point is from the boundary, so we can drop the sign.
abs_distances = [abs(pt) for pt in distances]

# Right now abs_distances is a list of 10000 numbers, and pcolormesh() needs a two dimensional list.
# We need to turn abs_distances into a 100 by 100 two dimensional array.
distances_matrix = np.reshape(abs_distances, (100, 100))

# Draw the heat map
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heatmap)  # put a legend on the heat map
# plt.show()
