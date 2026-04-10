from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./gpt2-finetuned")
model = AutoModelForCausalLM.from_pretrained("./gpt2-finetuned")

prompt = "What is an object?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=512,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))