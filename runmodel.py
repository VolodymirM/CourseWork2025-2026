from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./gpt2-finetuned")
model = AutoModelForCausalLM.from_pretrained("./gpt2-finetuned")

prompt = "Write a simple Java Spring program."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=100,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))