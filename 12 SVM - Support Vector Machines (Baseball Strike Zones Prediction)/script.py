import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
import pandas as pd

aaron_judge = pd.read_csv('aaron_judge.csv')
jose_altuve = pd.read_csv('jose_altuve.csv')
david_ortiz = pd.read_csv('david_ortiz.csv')

# The axes of our graph
fig, ax = plt.subplots()


# Create the labels

# let’s take a look at all of the features of a pitch
# print(aaron_judge.columns)
# print(aaron_judge.description.unique())

# We’re interested in looking at whether a pitch was a ball or a strike.
# print(aaron_judge.type.unique())

# We’ll want to use this feature as the label of our data points.
# However, instead of using strings, it will be easier if we change every 'S' to a 1 and every 'B' to a 0.
aaron_judge['type'] = aaron_judge['type'].map({'S': 1, 'B': 0})
# print(aaron_judge.type)


# Plotting the pitches

# We want to predict whether a pitch is a ball or a strike based on its location over the plate.
# A ball is a pitch thrown outside the strike zone. A strike is a pitch inside the strike zone.

# We can find the ball’s location in the columns plate_x and plate_z.

# plate_x measures how far left or right the pitch is from the center of home plate.
# If plate_x = 0, that means the pitch was directly in the middle of the home plate.

# plate_z measures how high off the ground the pitch was.
# If plate_z = 0, that means the pitch was at ground level when it got to the home plate.

# print(aaron_judge['plate_x'])

# We have the three columns we want to work with: 'plate_x', 'plate_z', and 'type'.
aaron_judge = aaron_judge.dropna(subset = ['plate_x', 'plate_z', 'type'])

plt.scatter(x = aaron_judge['plate_x'], y = aaron_judge['plate_z'], c = aaron_judge['type'], cmap = plt.cm.coolwarm, alpha = 0.5)
# plt.show()


# Building the SVM

# We want to validate our model, so we need to split the data into a training set and a validation set.
training_set, validation_set = train_test_split(aaron_judge, random_state = 1)

classifier = SVC(kernel = 'rbf')
X = training_set[['plate_x', 'plate_z']]
y = training_set['type']
classifier.fit(X, y)

# Visualize the SVM to see the predicted strike zone.
# draw_boundary(ax, classifier)
# plt.show()

print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))


# Optimizing the SVM

# Set the parameters of the SVM to be gamma = 100 and C = 100.
# This will overfit the data, but it will be a good place to start.
classifier = SVC(kernel = 'rbf', gamma = 100, C = 100)
classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

# draw_boundary(ax, classifier)
# plt.show()

print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))

# Let's try to find a configuration of gamma and C that greatly improves the accuracy.
classifier = SVC(kernel = 'rbf', gamma = 3, C = 1)
classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

# draw_boundary(ax, classifier)
# plt.show()

print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))


# Explore Other Players

# Let’s see how different players’ strike zones change.
# Aaron Judge is the tallest player in the MLB. Jose Altuve is the shortest player.
player = jose_altuve
# player = david_ortiz

player['type'] = player['type'].map({'S': 1, 'B': 0})
player = player.dropna(subset = ['plate_x', 'plate_z', 'type'])

plt.scatter(x = player['plate_x'], y = player['plate_z'], c = player['type'], cmap = plt.cm.coolwarm, alpha = 0.5)
training_set, validation_set = train_test_split(player, random_state = 1)
classifier = SVC(kernel = 'rbf', gamma = 3, C = 1)
classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

draw_boundary(ax, classifier)
# To compare the strike zones, we can force the axes to be the same.
ax.set_ylim(-2, 6)
ax.set_xlim(-3, 3)
plt.show()

print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))