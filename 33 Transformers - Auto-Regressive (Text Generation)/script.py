from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed

# checkpoint = "distilgpt2"
checkpoint = "gpt2"

# Instantiate the model and its corresponding tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
model = GPT2LMHeadModel.from_pretrained(checkpoint)

# Encode the text
text = "Let's turn this string into a PyTorch tensor of tokens."
pt_tokens = tokenizer.encode(text, return_tensors="pt")
print(pt_tokens)

# Decode tokens back into a full-text string
list_tokens = [1532, 345, 821, 3555, 428, 11, 345, 875, 9043, 502, 0]
decoded_tokens = tokenizer.decode(list_tokens)
print(decoded_tokens)

# Test the model's text generation capabilities with a few prompts
prompt = "Hello, my name is"
# prompt = "How would you"
inputs = tokenizer.encode(prompt, return_tensors="pt")
# `pad_token_id` tells the model to use the 'end of sequence' token to fill in the white space in each sequence
output = model.generate(inputs, max_length=75, num_beams = 1, do_sample = True, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0]))


# Changing Temperature
set_seed(10)  # ensure reproducible outputs in the text generation
prompt = "Artificial intelligence is"

def generate_text(prompt, temperature):
    input = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input, max_length=75,num_return_sequences=1, do_sample=True, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
    return f"\n---\n{tokenizer.decode(output[0]).strip()}\n---\n"

# Warmer (higher) values reduce the predictability of the model and can generate more creative outputs
high_temp = generate_text(prompt, 1.2)
# Cooler (lower) values instruct the model to select only the most likely completions, making its output more predictable
low_temp = generate_text(prompt, 0.3)
print(high_temp)
print(low_temp)


## Explore token selection strategies
set_seed(10)

# Greedy Search
prompt = "The capital of France is"
inputs = tokenizer.encode(prompt, return_tensors="pt")
greedy_outputs = model.generate(inputs, max_length=50, pad_token_id=tokenizer.eos_token_id)
print('greedy_outputs')
print(tokenizer.decode(greedy_outputs[0]))

# Beam Search
prompt = "The capital of France is"
inputs = tokenizer.encode(prompt, return_tensors="pt")
beam_outputs = model.generate(
    inputs,
    max_length=50,
    early_stopping=True,
    pad_token_id=tokenizer.eos_token_id,
    num_beams = 5  # performs beam search using 5 beams
)
print('beam_outputs')
print(tokenizer.decode(beam_outputs[0]))

# N-gram Penalty
prompt = "The capital of France is"
inputs = tokenizer.encode(prompt, return_tensors="pt")
ngram_outputs = model.generate(
    inputs,
    max_length=50,
    num_beams=5,
    early_stopping=True,
    pad_token_id=tokenizer.eos_token_id,
    no_repeat_ngram_size = 2  # set the n-gram penalty by passing our desired size of n
)
print('ngram_outputs')
print(tokenizer.decode(ngram_outputs[0]))

# Sampling
prompt = "The capital of France is"
inputs = tokenizer.encode(prompt, return_tensors="pt")
set_seed(10)
sample_outputs = model.generate(
    inputs,
    no_repeat_ngram_size=2,
    max_new_tokens=40,
    pad_token_id=tokenizer.eos_token_id,
    do_sample = True,  # implement sampling
    temperature = 0.6,  # turn down the temperature to prevent the least likely outputs
    top_k = 50  # instruct the model to choose from among the top 50 most likely completions
)
print('sample_outputs')
print(tokenizer.decode(sample_outputs[0]))
