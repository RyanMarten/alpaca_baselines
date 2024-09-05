# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

llama_path = "/work/10159/rmarten/ls6/dcft/llama-7b-checkpoints/"
tokenizer = AutoTokenizer.from_pretrained(llama_path)
model = AutoModelForCausalLM.from_pretrained(llama_path)

# Test the model
input_texts = ["Hello, how are you?", "Give me a list of the biggest cities in the world"]


for input_text in input_texts:
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generate a response
    output = model.generate(input_ids, max_length=1024)

    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"Input: {input_text}")
    print(f"Output: {generated_text}")

