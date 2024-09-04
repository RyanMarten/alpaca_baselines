import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

llama_path = "/work/10159/rmarten/ls6/dcft/llama-7b-checkpoints/"
tokenizer = AutoTokenizer.from_pretrained(llama_path)
model = AutoModelForCausalLM.from_pretrained(llama_path)


def generate(instruction):
    input_ids = tokenizer(instruction, return_tensors="pt").input_ids

    # Generate a response
    output = model.generate(input_ids, max_length=1024)

    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response


eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
for example in eval_set:
    # generate here is a placeholder for your models generations
    example["output"] = generate(example["instruction"])
    example["generator"] = "alpaca-7b-repro-10k-sft-gpt4o" # name of your model
