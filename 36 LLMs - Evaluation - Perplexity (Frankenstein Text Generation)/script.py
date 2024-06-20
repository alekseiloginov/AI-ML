import random
import torch
import pandas as pd
import transformers
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
set_seed()

# GPU version
# mistral7b = 'mistralai/Mistral-7B-v0.1'
# model_name = mistral7b

# CPU version
gpt2 = "distilbert/distilgpt2"
model_name = gpt2


# EDA
df = pd.read_csv("frankenstein_chunks.csv")
df.head()

print("Dataframe Info:")
print(df.info())
print("\n")
print("Dataframe Description:")
print(df.describe())
print("\n")
print("Number of unique values in each column:")
print(df.nunique())
random_index= random.randint(0, len(df) - 1)
print(df.loc[random_index, 'text'])
print(df.isnull().sum())
# df = df[:len(df)//2]  # CPU version

# convert this to a train/test split
train_df, test_df = train_test_split(df, test_size=0.2)

# convert the train_df and test_df from Pandas into Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


# Model Import and Tokenization

# GPU version:
# # 4-bit quantize the model
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
# # instantiate the model
# model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, token='YOUR_HUGGING_FACE_ACCESS_TOKEN')

# CPU version:
model = AutoModelForCausalLM.from_pretrained(model_name)


# check what it's running on
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cuda"
model.to(device)
print("\n\nModel is running on:" + "\n")
print(model.device)


# GPU version:
# # Prepare the model for QLoRA
# model = prepare_model_for_kbit_training(model)
# # Configure LoRA for our finetuning run
# config = LoraConfig(
#     r=32,
#     lora_alpha=64,
#     lora_dropout=0.05,
#     task_type="CAUSAL_LM"
# )
# model = get_peft_model(model, config)


# tokenize the data
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_train_dataset= train_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
tokenized_test_dataset = test_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)


# Base Model Evaluation
# generates next text given a starting prompt
def generate_text(prompt):
    print("prompt:", prompt)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "cuda"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=100)

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("output:", output)
    return output

# Generate a completion with the base model for informal evaluation.
base_generation = generate_text("I'm afraid I've created a ")
print(base_generation)

# calculated average perplexity of a model
def calc_perplexity(model):
    total_perplexity = 0
    for row in test_dataset:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # device = "cuda"

        # get input text + corresponding vectorized tokens to compare with model's generated outputs
        inputs = tokenizer(row['text'], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Calculate the loss without updating the model
        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)

        # get cross entropy for generated outputs
        loss = outputs.loss

        # exponentiate the cross-entropy to get perplexity
        perplexity = torch.exp(loss)
        total_perplexity += perplexity

    num_test_rows = len(test_dataset)
    avg_perplexity = total_perplexity / num_test_rows
    return avg_perplexity

base_ppl = calc_perplexity(model)
print(base_ppl)


# Training
tokenizer.pad_token = tokenizer.eos_token
model.config.use_cache = False

# GPU version
# trainer = transformers.Trainer(
#     model=model,
#     train_dataset=tokenized_train_dataset,
#     args=transformers.TrainingArguments(
#         warmup_steps=2,
#         fp16=True,
#         logging_steps=1,
#         save_steps=200,
#         output_dir="outputs",
#         per_device_train_batch_size=2,
#         num_train_epochs=2,
#         learning_rate=2e-5,
#         optim="paged_adamw_8bit"
#     ),
#     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
# )

# CPU version
trainer = transformers.Trainer(
    train_dataset=tokenized_train_dataset,
    model=model,
    args=transformers.TrainingArguments(
        # use_mps_device=True,  # this is done by default now if GPU is available
        warmup_steps=200,
        logging_steps=1,
        save_steps=200,
        output_dir="outputs",
        per_device_train_batch_size=2,
        num_train_epochs=2,
        learning_rate=2e-5,
        optim="adamw_hf"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Finetune the model
trainer.train()


# Evaluating the finetuned model
# generate a completion with the finetuned model and compare it to the base generation.
ft_generation = generate_text("I'm afraid I've created a ")
print("Base model generation: " + base_generation + "\n\n")
print("Finetuned generation: " + ft_generation)

# calculate the finetuned model's perplexity and compare it to the base model's.
ft_ppl = calc_perplexity(model)
print("Base model perplexity: " + str(base_ppl))
print("Finetuned model perplexity: " + str(ft_ppl))
