# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb

llama_path = "/work/10159/rmarten/ls6/dcft/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(llama_path)
model = AutoModelForCausalLM.from_pretrained(llama_path)
model.to('cuda')

while True:
    input_text = input("Enter your message (or 'quit' to exit): ")

    if input_text.lower() == 'quit':
        print("Exiting the program.")
        break

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    input_ids = input_ids.to('cuda')

    output = model.generate(input_ids, max_length=50)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Output: {generated_text}")
    print()
