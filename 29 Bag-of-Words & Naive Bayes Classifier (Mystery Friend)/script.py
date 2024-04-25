from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Implement Bag-of-Words (BoW) using CountVectorizer
bow_vectorizer = CountVectorizer()

# Vectorize the training and test texts so they can be used by a classifier
friends_docs = goldman_docs + henson_docs + wu_docs
friends_vectors = bow_vectorizer.fit_transform(friends_docs)

mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""
mystery_vector = bow_vectorizer.transform([mystery_postcard])

# Take a look at the friends' writing samples to get a sense of how they write.
print(goldman_docs[49])
print(henson_docs[49])
print(wu_docs[49])

# Implement a Naive Bayes classifier to predict which friend wrote the mystery card.
friends_classifier = MultinomialNB()

# Create labels with the size matching the number of vectors
friends_labels = ["Emma"] * 154 + ["Matthew"] * 141 + ["Tingfang"] * 166

# Train the classifier
friends_classifier.fit(friends_vectors, friends_labels)

# Predict the author of the `mystery_postcard`
predictions = friends_classifier.predict(mystery_vector)
mystery_friend = predictions[0] if predictions[0] else "someone else"
print("The postcard was from {}!".format(mystery_friend))

# Check the estimated probabilities that the `mystery_postcard` was written by each person
print(friends_classifier.predict_proba(mystery_vector))