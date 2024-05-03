RNNs - Recurrent Neural Networks -- LSTM - Long Short-Term Memory -- Seq2seq (Machine Translation)

The language pair datasets are from the Tatoeba Project: https://www.manythings.org/anki/.

Goals:
- Use Keras models with seq2seq neural networks to build a translation tool.
- Preprocess and tokenize the input and target texts.
- Use one-hot encoding to encode the text into tensors.
- Use Teacher Forcing and ground truth to facilitate faster learning.
- Use encoder-decoder model structure to encode the input text into state vectors and then decode it into another language.
- Train the model on language pairs and use it for machine translation of unknown sentences.
