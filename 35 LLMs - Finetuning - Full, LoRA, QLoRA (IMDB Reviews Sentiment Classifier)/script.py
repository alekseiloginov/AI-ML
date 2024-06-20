import torch
import random
import pandas as pd
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, LoftQConfig

# set a random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
set_seed()

## 1. Perform EDA (Exploratory Data Analysis)
df = pd.read_csv('imdb_data.csv')
print(df.head())

print("Number of null values:")
print(df.isnull().sum())

print("Dataframe Info:")
print(df.info())
print("\n")

print("Dataframe Description:")
print(df.describe())
print("\n")

print("Number of unique values in each column:")
print(df.nunique())

# look at a random review to get a feel for how they sound
random_index = random.randint(0, len(df) - 1)
pprint(df.loc[random_index, 'text'])

# tokenize reviews
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # this tokenizer will work for our smaller model
tokenized_reviews = df['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# check reviews length in tokens
review_token_lengths = tokenized_reviews.apply(len)
print(f"Shortest review length (in tokens): {review_token_lengths.min()}")
print(f"Longest review length (in tokens): {review_token_lengths.max()}")
print(f"Average review length (in tokens): {review_token_lengths.mean()}")
# Some of the examples in the dataset are longer than our model can intake.
# We'll need to be careful about how we handle the longer reviews in our dataset.


## 2. Evaluate the base model on our IMDB dataset
# convert a DataFrame into a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# tokenize the text
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="longest", truncation=True)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# instantiate our model, a tiny version of BERT
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

training_args = TrainingArguments(
    output_dir='./temp_results',
    do_train=False,  # tell the model it's in 'evaluation mode' (and not 'training mode')
    do_eval=True,
    seed=42
)

# filter our dataset for only 'test' data
eval_dataset = tokenized_dataset.filter(lambda x: x['dataset'] == 'test')

# put everything together in Hugging Face's `Trainer` class
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset
)

# evaluate the model's results
eval_results = trainer.evaluate()
print("No finetuning results:", eval_results)
# 'eval_loss': 0.6899291276931763 - we'll be comparing `eval_loss` number to the finetuned models later on


## 3. A Full Finetune of BERT with IMDB Data
# convert the imported DataFrame into a Hugging Face Dataset
full_dataset = Dataset.from_pandas(df)

# split it into training and test sets using the 'dataset' column in the original data
training_set = full_dataset.filter(lambda example: example['dataset'] == 'train')
test_set = full_dataset.filter(lambda example: example['dataset'] == 'test')

# tokenize the text
tokenized_training_set = training_set.map(tokenize_function, batched=True)
tokenized_test_set = test_set.map(tokenize_function, batched=True)

# define training arguments
training_args = TrainingArguments(
    output_dir="./temp_results",  # where the model's output is saved
    warmup_steps=500,  # length of the warm up phase at the start of training. Gradually increasing the learning rate at the start of training can help the model avoid bad outcomes early in the training process.
    weight_decay=0.01,  # helps prevent overfitting by reducing the magnitude of the model's weights
    logging_dir="./logs",  # where to save the training logs
    logging_steps=10,
    learning_rate=1e-4,  # size of the steps the optimizer takes for each iteration of gradient descent
    save_strategy="no",  # how we wish to save checkpoints of the model across different epochs
    num_train_epochs=3,  # e
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
)

# put everything together in Hugging Face's `Trainer` class
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_training_set,
    eval_dataset=tokenized_test_set
)

# perform our finetuning run
trainer.train()
# {'train_runtime': 24.7984, 'train_samples_per_second': 151.219, 'train_steps_per_second': 12.702, 'train_loss': 0.6572624085441469, 'epoch': 3.0}

# evaluate the model's results
print("Full finetuning results:", trainer.evaluate())
# 'eval_loss': 0.5894039273262024 (improvement from 0.6899291276931763 we had above without finetuning)

# save the model so we can use it on our own
model.save_pretrained("./full_finetuned_model")


## 4. Using our finetuned model for sentiment classification
# load the model from disk
model_path = "./full_finetuned_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# define a pipeline with a model and a tokenizer
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# write a short movie review
single_text = "It's a good movie"

# run classifier pipeline
result = classifier(single_text)
print(f"Prediction: {result[0]['label']}, Score: {result[0]['score']}")

## run multiple movie reviews in batches
batch_texts = ["It's a good movie", "This movie is not bad", "Not worth it"]
results = classifier(batch_texts)
for i, result in enumerate(results):
    print(f"Prediction for text {i+1}: {result['label']}, Score: {result['score']}")


## 5. Finetuning BERT with LoRA
# instantiate a tiny version of BERT model
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

# configure our LoRA model
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # Text classification
    inference_mode=False,  # false in training, true when inferring
    r=16,  # Rank of low-rank matrices that parameterize the LoRA layers
    lora_alpha=32,  # importance of LoRA vs. original weights (typically about double the size of the rank)
    lora_dropout=0.1  # Dropout rate, helps prevent overfitting
)

# create our LoRA model (the original model + a LoRA adapter added to it)
lora_model = get_peft_model(model, peft_config)

# Define a function to show the total number of parameters in the base model,
# the number of parameters we'll train using LoRA,
# and the compression ratio of the number of LoRA parameters to the number of original model parameters.
# This will help us to see how much we've lowered the compute required for this finetune.
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():  # named parameters are weights and biases of a model
        all_param += param.numel()  # numel is the total number of elements a parameter (tensor) contains
        if param.requires_grad:  # if the parameter requires gradient (trainable)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || compression ratio (% of trainable params): {100 * trainable_params / all_param}"
    )

print_trainable_parameters(lora_model)

# move our LoRA model to GPU to see if we can speed up our finetuning
print("LoRA model's device BEFORE moving to GPU:", lora_model.device)
# {'train_runtime': 11.9867, 'train_samples_per_second': 312.846, 'train_steps_per_second': 26.279, 'train_loss': 0.6930862184554811, 'epoch': 3.0}

device = torch.device("mps")  # use Apple silicon mac's GPU
# device = torch.device("cuda")  # use NVIDIA's CUDA GPU
lora_model.to(device)  # as of now, we only need to move the model, as Hugging Face's Dataset already moves the data to GPU if it's available

print("LoRA model's device AFTER moving to GPU:", lora_model.device)
# {'train_runtime': 11.9189, 'train_samples_per_second': 314.627, 'train_steps_per_second': 26.429, 'train_loss': 0.6930862184554811, 'epoch': 3.0}

# put everything together in Hugging Face's `Trainer` class
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_training_set,
    eval_dataset=tokenized_test_set
)

# perform finetuning
trainer.train()

# evaluate the model's results
print("LoRA finetuning results:", trainer.evaluate())

# save this model for future use
model.save_pretrained("./lora_finetuned_model")


## 6. Finetuning BERT with QLoRA
# we'll now quantize our BERT model before performing a LoRA finetune on it

# define the config of our quantization library BitsAndBytes
# There are several options for how many bits you'd like to quantize the model to, with 8 a common configuration.
# To keep things simple, we'll quantize our BERT model to 4 bits.
config = BitsAndBytesConfig(
    load_in_4bit=True,  # load the model in 4-bit precision
    bnb_4bit_quant_type="nf4",  # type of quantization: use BitsAndBytes (BnB)'s "nf4" (normalized float 4-bit)
    bnb_4bit_use_double_quant=True,  # use double quantization to help reduce the quantization error
    bnb_4bit_compute_dtype=torch.bfloat16  # data type for computation: 16-bit floating point format (precision of weights for computation, when they're temporarily not quantized)
)

# prepare our model for quantization-aware training using QLoRA
# This time, when we instantiate our BERT model, we'll pass the quantization configuration.
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2, quantization_config=config)

# prepare our model for quantization
model = prepare_model_for_kbit_training(model)  # `kbit` here refers to quantizing the model to a certain number (k) of bits

# initialize LoftQConfig - use Hugging Face's LoftQ library for QLoRA that quantizes the model with LoRA finetuning in mind
loftq_config = LoftQConfig(loftq_bits=4)  # specify that we want to quantize our model to 4 bits

# configure our QLoRA model
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    loftq_config=loftq_config,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)

# instantiate the QLoRA model
qlora_model = get_peft_model(model, peft_config)

# print the trainable parameters
print_trainable_parameters(qlora_model)

# set up our trainer
trainer = Trainer(
    model=qlora_model,
    args=training_args,
    train_dataset=tokenized_training_set,
    eval_dataset=tokenized_test_set
)

# perform finetuning
trainer.train()

# evaluate the model's results
print("QLoRA finetuning results:", trainer.evaluate())
