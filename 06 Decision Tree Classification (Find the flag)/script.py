import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#dataset source: https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
        'red','green','blue','gold','white','black','orange','mainhue','circles',
        'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
dataset = pd.read_csv("flag.csv", names = cols)

#variable names to use as predictors
predictors = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue','bars','stripes', 'circles',
               'crosses', 'saltires','quarters','sunstars','triangle','animate']

#Print number of countries by landmass, or continent
print(dataset['landmass'].value_counts())

#Create a new dataframe with only flags from Europe and Oceania
europe_oceania = dataset[dataset['landmass'].isin([3,6])]
print(europe_oceania)

#Print the average vales of the predictors for Europe and Oceania
numeric_predictors = [p for p in predictors if dataset[p].dtype != 'object']
print(europe_oceania.groupby('landmass')[numeric_predictors].mean().T)

#Create labels for only Europe and Oceania
labels = europe_oceania['landmass']
print(labels)

#Print the variable types for the predictors
print(europe_oceania[predictors].dtypes)

#Create dummy variables for categorical predictors
transformed_predictors = pd.get_dummies(europe_oceania[predictors])
print(transformed_predictors)

#Split data into a train and test set
x_train, x_test, y_train, y_test = train_test_split(
        transformed_predictors, labels, random_state=1, test_size=.4)

#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []
for depth in depths:
    dtree_classifier = DecisionTreeClassifier(max_depth=depth, random_state = 10)
    dtree_classifier.fit(x_train, y_train)
    score = dtree_classifier.score(x_test, y_test)
    acc_depth.append(score)
    print(depth, score)

#Plot the accuracy vs depth
plt.plot(depths, acc_depth)
plt.xlabel('max depths')
plt.ylabel('accuracy')
plt.show()

#Find the largest accuracy and the depth this occurs
max_accuracy = np.max(acc_depth)
# max_acc_depth = depths[acc_depth.index(max_accuracy)]
max_accu_depth = depths[np.argmax(acc_depth)]
print(f'Highest accuracy {round(max_accuracy,3)*100}% at depth {max_accu_depth}')

#Refit decision tree model with the highest accuracy and plot the decision tree
dtree_classifier = DecisionTreeClassifier(max_depth=max_accu_depth, random_state = 1)
dtree_classifier.fit(x_train, y_train)
tree.plot_tree(dtree_classifier,
               feature_names=x_train.columns.tolist(),
               class_names=['Europe','Oceania'],
               filled=True)
plt.show()

#Create a new list for the accuracy values of a pruned decision tree.
#Loop through the values of ccp and append the scores to the list.
acc_pruned = []
ccp = np.logspace(-3, 0, num=20)  # generate 20 numbers from 0 to 1
for i in ccp:
    dt_prune = DecisionTreeClassifier(ccp_alpha = i,
                                      max_depth = max_accu_depth,
                                      random_state = 1)
    dt_prune.fit(x_train, y_train)
    prune_score = dt_prune.score(x_test, y_test)
    acc_pruned.append(prune_score)
    print(i, prune_score)

plt.plot(ccp, acc_pruned)
plt.xscale('log')
plt.xlabel('CCP alpha')
plt.ylabel('accuracy')
plt.show()

#Find the largest accuracy and the ccp value this occurs
max_acc_pruned = np.max(acc_pruned)
best_ccp = ccp[np.argmax(acc_pruned)]

print(f'Highest accuracy {round(max_acc_pruned,3)*100}% at ccp_alpha {round(best_ccp,4)}')

#Fit a decision tree model with the best max_depth and ccp_alpha found above
dt_final = DecisionTreeClassifier(random_state = 1,
                                  max_depth = max_accu_depth,
                                  ccp_alpha=best_ccp)
dt_final.fit(x_train, y_train)

#Plot the final decision tree
plt.figure(figsize=(14,8))
tree.plot_tree(dt_final,
               feature_names = x_train.columns.tolist(),
               class_names = ['Europe', 'Oceania'],
               filled=True)
plt.show()
