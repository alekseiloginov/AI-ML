Transformers - Finetuning - Full, LoRA, QLoRA (IMDB Reviews Sentiment Classifier)

IMDB movie reviews dataset is from Hugging Face: https://huggingface.co/datasets/imdb.
In the dataset, `text` column holds the reviews, `label` is 1 is positive sentiment, 0 is negative sentiment. 
The `dataset` column separates the data into training and test sets.

Goals:
- Use LoRA and QLoRA to finetune LLMs.
- Perform EDA (Exploratory Data Analysis)
- Evaluate the base BERT model on the IMDB dataset
- Perform a Full Finetuning of BERT with IMDB Data
- Use the finetuned model for sentiment classification
- Finetune BERT with LoRA
- Finetune BERT with QLoRA
