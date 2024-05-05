import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint = "google/flan-t5-small"
# checkpoint = "google/flan-t5-base"
# checkpoint = "facebook/bart-large"

# Text to be summarized
text_example = "Large language models (LLMs) have a dirty secret: they require vast amounts of energy to train and run. What’s more, it’s still a bit of a mystery exactly how big these models’ carbon footprints really are. AI startup Hugging Face believes it’s come up with a new, better way to calculate that more precisely, by estimating emissions produced during the model’s whole life cycle rather than just during training. It could be a step toward more realistic data from tech companies about the carbon footprint of their AI products at a time when experts are calling for the sector to do a better job of evaluating AI’s environmental impact. Hugging Face’s work is published in a non-peer-reviewed paper. To test its new approach, Hugging Face estimated the overall emissions for its own large language model, BLOOM, which was launched earlier this year. It was a process that involved adding up lots of different numbers: the amount of energy used to train the model on a supercomputer, the energy needed to manufacture the supercomputer’s hardware and maintain its computing infrastructure, and the energy used to run BLOOM once it had been deployed. The researchers calculated that final part using a software tool called CodeCarbon, which tracked the carbon dioxide emissions BLOOM was producing in real time over a period of 18 days. Hugging Face estimated that BLOOM’s training led to 25 metric tons of carbon dioxide emissions. But, the researchers found, that figure doubled when they took into account the emissions produced by the manufacturing of the computer equipment used for training, the broader computing infrastructure, and the energy required to actually run BLOOM once it was trained."
print(text_example)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Tokenize the text
# "summarize: " prefix prompts the T5 model that this is a summarization task
summarization_input = "summarize: " + text_example
tokens_input = tokenizer.encode(summarization_input, return_tensors = "pt", truncation = True)
print(np.shape(tokens_input))

# Perform summarization
summary_ids = model.generate(tokens_input, min_length=80, max_length=120)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
