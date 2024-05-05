Transformers - Auto-Encoding (Sentiment Analysis)

`distilbert-base-uncased-finetuned-sst-2-english` Large Language Model (encoder-only) is from Hugging Face:
https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english.
This is a smaller version of “BERT”, a bidirectional encoder-only transformer model. 
This specific version was finetuned for sentiment classification.

More specifically, `distilbert-base-uncased-finetuned-sst-2-english` is a version of `distilbert-base-uncased` 
(a distilled version of BERT `bert-base-uncased` that is smaller and faster), 
that has been finetuned on the SST-2 dataset (Stanford Sentiment Treebank corpus).

Goals:
- Explore Hugging Face's transformers package to implement language tasks such as sentiment analysis.
- Perform text preprocessing and tokenization prior to using transformers.
- Choose the appropriate tokenizer and model configurations.
- Explore batching, padding and truncation used during tokenization.
- Perform different language tasks with encoder-only models.
- Use the pipeline() object for model inference.
