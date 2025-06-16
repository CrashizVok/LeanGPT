import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer)

MODEL_NAME = "gpt2" 
DATA_PATH = "data.json"
OUTPUT_DIR = "./LeanGPT"  
BATCH_SIZE = 6
EPOCHS = 3  
MAX_LENGTH = 384  
LOGGING_STEPS = 50

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

dataset = load_dataset("json", data_files=DATA_PATH)["train"]

def preprocess_data(examples):
    texts = [f"User: {inp}\nBot: {resp}" for inp, resp in zip(examples["input"], examples["response"])]

    tokenized_inputs = tokenizer(
        texts, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )

    labels = tokenized_inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100  

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(preprocess_data, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_strategy="epoch", 
    evaluation_strategy="no",  
    logging_dir="./logs",
    logging_steps=LOGGING_STEPS,
    fp16=torch.cuda.is_available(), 
    learning_rate=3e-5,  
    warmup_steps=200,  
    weight_decay=0.01, 
    gradient_accumulation_steps=2,  
    save_total_limit=5,  
    push_to_hub=False  
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"{OUTPUT_DIR}")
