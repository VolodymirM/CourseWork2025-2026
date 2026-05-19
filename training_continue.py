import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(script_dir, "gpt2-finetuned", "checkpoint-7200")

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

script_dir = os.path.dirname(os.path.abspath(__file__))
trainingdata_dir = os.path.join(script_dir, "trainingdata")

extensions = [
    ".data",
    ".html", ".css", ".java", ".ts", ".js", ".json", ".xml"
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
    return tokenizer(example["text"], truncation=False)  # no padding yet

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Pack tokens into 512-token blocks instead of padding each line
block_size = 512

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
              for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = tokenized_datasets.map(group_texts, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    num_train_epochs=5,
    eval_strategy="steps",
    eval_steps=200,
    load_best_model_at_end=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    warmup_steps=480,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    fp16=True,
    lr_scheduler_type="cosine"
)

split = tokenized_datasets["train"].train_test_split(test_size=0.05, seed=42)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=checkpoint_path)
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")