import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_and_test_model(model_path):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Move model to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"The model is on device: {model.device}")

    # Test the model
    input_text = "Hello, how are you?"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    print(f"The input_ids are on device: {input_ids.device}")

    # Generate a response
    output = model.generate(input_ids, max_length=50)

    # Decode the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Input: {input_text}")
    print(f"Output: {generated_text}")

def main():
    parser = argparse.ArgumentParser(description="Load and test a LLaMA model.")
    parser.add_argument("model_path", type=str, help="Path to the LLaMA model")
    args = parser.parse_args()

    load_and_test_model(args.model_path)

if __name__ == "__main__":
    main()
