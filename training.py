import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

script_dir = os.path.dirname(os.path.abspath(__file__))
trainingdata_dir = os.path.join(script_dir, "trainingdata")
train_files = [os.path.join(trainingdata_dir, f) for f in os.listdir(trainingdata_dir) if f.endswith(".data")]
dataset = load_dataset("text", data_files={"train": train_files})

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=100,
    save_total_limit=1,
    logging_steps=20,
    report_to="none",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")