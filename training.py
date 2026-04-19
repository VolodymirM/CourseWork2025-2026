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

extensions = [
    ".data", ".html", ".css", ".java", ".ts", ".js", ".json", ".xml"
]

train_files = []
for root, _, files in os.walk(trainingdata_dir):
    for f in files:
        if any(f.endswith(ext) for ext in extensions):
            file_path = os.path.join(root, f)
            try:
                if os.path.getsize(file_path) > 0:
                    # Try to open and read a small chunk to ensure it's valid UTF-8
                    with open(file_path, 'r', encoding='utf-8') as test_file:
                        test_file.read(1024)
                    train_files.append(file_path)
            except Exception:
                pass

if not train_files:
    raise RuntimeError("No valid UTF-8 training files found.")

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
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    fp16=True,
    lr_scheduler_type="cosine"
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