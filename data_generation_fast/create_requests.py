import json
import random
import re

def encode_prompt(prompt_instructions, prompt_file='../prompt.txt'):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(prompt_file).read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt

def create_instruction_requests(
    output_file="./requests_to_parallel_process.jsonl",
    seed_tasks_path="../seed_tasks.jsonl",
    num_instructions_to_generate=100,
    model_name="gpt-4o-2024-08-06",
    num_prompt_instructions=3):

    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    # each request creates ~16 new instructions (19 - 3 in context), let's say 14 on average to be safe and round up
    n_requests = (num_instructions_to_generate // 14) + 1
    jobs = []

    print(f"Creating requests, each with {num_prompt_instructions} sampled in-context example seed instructions")
    for i in range(n_requests):
        # for each request, sample a different prompt
        seed_idxs = random.sample(range(len(seed_instruction_data)), num_prompt_instructions)
        prompt_instructions = [seed_instruction_data[i] for i in seed_idxs]
        prompt = encode_prompt(prompt_instructions)

        job = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "logit_bias": {"199999": -100},
            "stop": ["\n20", "20.", "20."],
            "max_tokens": 3072,
            "metadata": { 
                "request_idx" : i,
                "seed_idxs": seed_idxs
            }
        }
        jobs.append(job)

    print(f"Writing jobs to {output_file}")
    with open(output_file, "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")

if __name__ == "__main__":
    create_instruction_requests()