# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb

llama_path = "/work/10159/rmarten/ls6/dcft/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(llama_path)
model = AutoModelForCausalLM.from_pretrained(llama_path)

model.to('cuda')

print(f"The model is on device: {model.device}")

# Test the model
input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
input_ids = input_ids.to('cuda')
print(f"The input_ids are on device: {input_ids.device}")

# Generate a response
output = model.generate(input_ids, max_length=50)

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Input: {input_text}")
print(f"Output: {generated_text}")

