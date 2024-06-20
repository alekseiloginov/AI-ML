LLMs - Evaluation - Perplexity (Frankenstein Text Generation)

The original text of Shelley's Frankenstein is from Project Gutenberg: https://www.gutenberg.org/ebooks/84.

Models we're using are from Hugging Face:
- Mistral 7B: https://huggingface.co/mistralai/Mistral-7B-v0.1
- Distil GPT-2: https://huggingface.co/distilbert/distilgpt2

Goals:
- Finetune a custom model that can write original Frankenstein.
- We have two options: either perform QLoRA on the Mistral 7B language model, or perform a full finetune on Distil GPT-2.
- Perform EDA on a dataset made of snippets from Frankenstein.
- Convert data from Pandas into Hugging Face Datasets and tokenize the data.
- For QLoRA: Shrink a generative language model with 4bit quantization; and configure LoRA to train a small subset of the quantized model's parameters.
- Configure and train the finetuned model.
- Evaluate and compare the base model and the finetuned model using both perplexity and more informal methods.
