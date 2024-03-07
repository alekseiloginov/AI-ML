from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Exploring the Data

emails = fetch_20newsgroups()
# print(emails.target_names)

# We’re interested in seeing how effective our Naive Bayes classifier is at telling the difference between a baseball email and a hockey email.
emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'])
# print(emails.data[5])
# print(emails.target_names[emails.target[5]])


# Making the Training and Test Sets

train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)
test_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)


# Counting Words

counter = CountVectorizer()

# Tell counter what possible words can exist in our emails.
counter.fit(test_emails.data + train_emails.data)

# Make a list of the counts of our words in our training and test sets.
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)


# Making a Naive Bayes Classifier

classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)
print(classifier.score(test_counts, test_emails.target))


# Testing Other Datasets

# Let's see how the classifier does with emails about really different topics.
train_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)
test_emails = fetch_20newsgroups(categories = ['comp.sys.ibm.pc.hardware', 'rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)

counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)
train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

classifier = MultinomialNB()
classifier.fit(train_counts, train_emails.target)
print(classifier.score(test_counts, test_emails.target))