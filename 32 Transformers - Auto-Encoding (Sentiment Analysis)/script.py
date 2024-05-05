import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig, BertModel, AutoTokenizer, pipeline

# Set the model checkpoint
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

## Sentiment analysis with AutoModelForSequenceClassification2
# Preprocess input with a tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = ["I've been waiting to learn about transformers my whole life.",
              "I hate this so much!"]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt") # `pt` or `tf` = PyTorch or TensorFlow
print(inputs)

# Use AutoModelForSequenceClassification model head to perform sentiment analysis
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)  # double asterisk ensures that the inputs are stored as a dictionary within the function
print(outputs.logits.shape)
print(outputs.logits)

# Convert the tensor output to a probability distribution
# Transformer models returns logits - raw, un-normalized scores outputted by the last layer of the model.
# They need to be passed through a softmax function.
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)  # returns matrix with corresponding probabilities of a sentence belonging to either class
print(model.config.id2label)  # returns class labels for classification


## Load a transformer model architecture
# When no model is specified, AutoModel will automatically guess the appropriate model architecture for a specified task.
# If we already know the kind of model we'd like to use, say BERT (as in the previous section), we can load it directly.
config = BertConfig()
print(config)
model = BertModel(config)
print(model)
# We've loaded the architecture of BERT and randomly initialized its weights.
# It is, however, fairly useless for inference at the moment as it hasn't been trained yet.

# Load a transformer model that is already trained using the from_pretrained() method
model = BertModel.from_pretrained("bert-base-cased")
# This model is initialized with all the weights of the checkpoint.
# Additionally, the weights are downloaded and cached so future calls to the checkpoint won't re-download them.

# Creat a tokenizer from a model checkpoint
sample_text = ["BERT is an encoder-only transformer."]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print(tokenizer(sample_text))


## Explore tokenization
# Break text into words using Python's split() function
sample_sentence = "Brevity is the soul of wit."
tokenized_array = sample_sentence.split()
print(tokenized_array)

# Tokenize text with WordPiece tokenizer used in BERT
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_array_wordpiece = tokenizer.tokenize(sample_sentence)
print(tokenized_array_wordpiece)

# Encode text (tokenize and vectorize)
sequence = "I've been waiting to explore HuggingFace my whole life."
input_tokens = tokenizer(sequence)
print(input_tokens)

# Decode text: from a tensor to a string
decoded_string = tokenizer.decode(input_tokens['input_ids'])
print(decoded_string)


## Explore batching, padding and truncation used during tokenization
# Batching
sequences = ["I've been waiting to explore HuggingFace my whole life.", "So have I!"]
model_inputs_batched = tokenizer(sequences)
print(model_inputs_batched)

# Padding
# Pad the sequences up to the maximum sequence length in the given batch
model_inputs_padded_1 = tokenizer(sequences, padding="longest")
print(model_inputs_padded_1)
# Pad the sequences up to the model max length (512 for BERT or DistilBERT)
model_inputs_padded_2 = tokenizer(sequences, padding="max_length")
print(model_inputs_padded_1)
# Pad the sequences up to the specified max length
model_inputs_padded_3 = tokenizer(sequences, padding="max_length", max_length=8)
print(model_inputs_padded_3)

# Truncation
sequences = ["I've been waiting to explore HuggingFace my whole life.", "So have I!"]
# Truncate the sequences that are longer than the model max length (512 for BERT or DistilBERT)
model_inputs_truncated_1 = tokenizer(sequences, truncation=True)
print(model_inputs_truncated_1)
# Truncate the sequences that are longer than the specified max length
model_inputs_truncated_2 = tokenizer(sequences, max_length=8, truncation=True)
print(model_inputs_truncated_2)


## Set up a sentiment analysis classifier using pipeline() function
classifier = pipeline(task = "sentiment-analysis",
                      model = checkpoint)
print(classifier)
sample_text = ["I've been waiting to learn about transformers my whole life.",
               "I hate this so much!"]
print(classifier(sample_text))