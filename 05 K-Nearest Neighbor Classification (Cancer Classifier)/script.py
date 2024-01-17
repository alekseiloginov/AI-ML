from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

## Explore the data
breast_cancer_data = load_breast_cancer()
# print(type(breast_cancer_data))

print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

## Splitting the data into Training and Validation Sets
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

print(len(breast_cancer_data.data))
print(len(training_data))
print(len(training_labels))

## Running the classifier
classifier = KNeighborsClassifier(n_neighbors = 3)

classifier.fit(training_data, training_labels)

print(classifier.score(training_data, training_labels))
print(classifier.score(validation_data, validation_labels))

## Finding the best k
accuracies = []
for k in range(1, 101):
    #Create classifier
    classifier = KNeighborsClassifier(n_neighbors = k)
    #Train classifier
    classifier.fit(training_data, training_labels)
    #Report accuracy
    score = classifier.score(validation_data, validation_labels)
    print(k, score)
    accuracies.append(score)

## Graphing the results
k_list = range(1, 101)
plt.plot(k_list, accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()