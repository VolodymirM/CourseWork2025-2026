from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationMixin

tokenizer = AutoTokenizer.from_pretrained("./gpt2-finetuned-trained-v2")
model = AutoModelForCausalLM.from_pretrained("./gpt2-finetuned-trained-v2")

prompt = (""" """)
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate( # type: ignore[reportAttributeAccessIssue]
    **inputs,
    max_new_tokens=256,
    do_sample=True,          # test with True
    temperature=0.4,      # slightly lower for more focus
    top_k=40,             # lower for more determinism
    top_p=0.90,           # slightly lower for less randomness
    repetition_penalty=1.15,  # add this to reduce repeated lines
    pad_token_id=tokenizer.eos_token_id,
)

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(result)