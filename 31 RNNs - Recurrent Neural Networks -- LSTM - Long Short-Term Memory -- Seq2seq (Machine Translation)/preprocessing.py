import numpy as np
import re

# Importing our translations
data_path = "eng-rus.txt"

# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

# Building empty lists to hold sentences
input_docs = []
target_docs = []
# Building empty vocabulary sets
input_tokens = set()
target_tokens = set()

# Regex pattern for splitting the text into tokens
word_or_punctuation = r"[\w']+|[^\s\w]"

# Choose the number of lines for preprocessing
for line in lines[:500]:
    # Input and target sentences are separated by tabs
    input_doc, target_doc = line.split('\t')[:2]
    # print(input_doc)
    # print(target_doc)
    # print('----------')

    # Appending each input sentence to input_docs
    input_docs.append(input_doc)

    # Split sentences into words and punctuation tokens separated by whitespaces.
    target_doc = " ".join(re.findall(word_or_punctuation, target_doc))
    # Redefine target_doc and append it to target_docs:
    target_doc = '<START> ' + target_doc + ' <END>'
    target_docs.append(target_doc)

    # Now we split up each sentence into words
    # and add each unique word to our vocabulary set
    for token in re.findall(word_or_punctuation, input_doc):
        # print(token)
        if token not in input_tokens:
            input_tokens.add(token)

    for token in target_doc.split():
        # print(token)
        if token not in target_tokens:
            target_tokens.add(token)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

# Create `num_encoder_tokens` and `num_decoder_tokens` representing length of each vocabulary for one-hot encoding
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

# Find longest sequence of tokens out of all sentences
max_encoder_seq_length = max([len(re.findall(word_or_punctuation, input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(word_or_punctuation, target_doc)) for target_doc in target_docs])

# Create direct and reverse lookup dictionaries
input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])

reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())

# Create 3D tensors to store input and target data for our training model
encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
# We will use Teacher Forcing, so we will need additional input for decoder with ground truth
decoder_input_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# Do one-hot encoding
for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):

    for timestep, token in enumerate(re.findall(word_or_punctuation, input_doc)):
        # Assign 1 for the current line, timestep, & word
        encoder_input_data[line, timestep, input_features_dict[token]] = 1.

    for timestep, token in enumerate(target_doc.split()):
        # Prepare ground truth
        decoder_input_data[line, timestep, target_features_dict[token]] = 1.

        if timestep > 0:
            # shift decoder's target data one timestep left (`timestep-1`), as it is one step ahead of its input data
            decoder_target_data[line, timestep-1, target_features_dict[token]] = 1.
