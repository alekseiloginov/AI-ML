import os
import gensim
import spacy
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words

# Preprocessing the Data

# get list of all speech files
files = sorted([file for file in os.listdir("inaugural") if file[-4:] == '.txt'])
# print(files)

# read each speech file
speeches = [read_file(file) for file in files]
# print(speeches)

# split each speech into tokens:
# - processed_speeches[0] represents the first inaugural address in processed_speeches.
# - processed_speeches[0][0] represents the first sentence in the first inaugural address in processed_speeches.
# - processed_speeches[0][0][0] represents the first word in the first sentence in the first inaugural address in processed_speeches.
processed_speeches = process_speeches(speeches)
# print(processed_speeches)

# merge all speeches at the top level
# to have only sentences and words
# (to build a custom set of word embeddings using `gensim`, we need to convert our data into a list of lists)
all_sentences = merge_speeches(processed_speeches)
# print(all_sentences)


# All Presidents

# view most frequently used words across all the inaugural addresses
most_freq_words = most_frequent_words(all_sentences)
# print(most_freq_words)

# create gensim model of all speeches
all_prez_embeddings = gensim.models.Word2Vec(all_sentences, vector_size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom
similar_to_freedom = all_prez_embeddings.wv.most_similar("freedom", topn=20)
# print(similar_to_freedom)


# One President

# get President Roosevelt sentences
roosevelt_sentences = get_president_sentences("franklin-d-roosevelt")
# print(roosevelt_sentences)

# view most frequently used words of Roosevelt
roosevelt_most_freq_words = most_frequent_words(roosevelt_sentences)
# print(roosevelt_most_freq_words)

# create gensim model for Roosevelt
roosevelt_embeddings = gensim.models.Word2Vec(roosevelt_sentences, vector_size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for Roosevelt
roosevelt_similar_to_freedom = roosevelt_embeddings.wv.most_similar("freedom", topn=20)
# print(roosevelt_similar_to_freedom)

# The results are less than satisfying (many of the words with similar embeddings are stop words, or commonly used words that give little insight into topic or context).
# This is because even for President Roosevelt, who gave the largest number of inaugural addresses, there is not enough data to produce robust word embeddings.


# Selection of Presidents

# Let's look at the speeches of multiple presidents to increase our corpus size and produce better word embeddings.

# get sentences of multiple presidents
# (The four presidents featured on Mount Rushmore in Keystone, South Dakota are George Washington, Thomas Jefferson, Theodore Roosevelt, and Abraham Lincoln)
rushmore_prez_sentences = get_presidents_sentences(["washington","jefferson","lincoln","theodore-roosevelt"])

# view most frequently used words of presidents
rushmore_most_freq_words = most_frequent_words(rushmore_prez_sentences)
# print(rushmore_most_freq_words)

# create gensim model for the presidents
rushmore_embeddings = gensim.models.Word2Vec(rushmore_prez_sentences, vector_size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for presidents
rushmore_similar_to_freedom = rushmore_embeddings.wv.most_similar("freedom", topn=20)
# print(rushmore_similar_to_freedom)
