import json
import os
import pdb
import re 
import string
from rouge_score import rouge_scorer
import tqdm
import multiprocessing
from functools import partial
import numpy as np
import time
import json
import fire

def response_iterator(responses_filepath):
    with open(responses_filepath) as file_in:
        for response in file_in:
            try:
                response_json = json.loads(response)
                prompt, completions, metadata = response_json

                if isinstance(completions, list) and "error" in completions[0]:
                    print(f"WARNING: request had an error {completions[0]}, skipping")
                    continue
                
                message = completions['choices'][0]['message']['content']
                finish_reason = completions['choices'][0]['finish_reason']
                
                if finish_reason != "stop":
                    print(f"WARNING: finish reason {finish_reason} for request {metadata}")
                
                yield message, metadata, finish_reason

            except Exception as e:
                print(f"Error processing response: {e}")
                continue

def post_process_response(num_prompt_instructions, message, finish_reason):
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + message
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and finish_reason == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions

def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)

def load_seed_instructions(seed_tasks_path="../seed_tasks.jsonl"):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
    seed_instructions = [d["instruction"] for d in seed_instruction_data]
    return seed_instructions

def append_to_json(file_path, new_data):
    # Read the existing data
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, start with an empty list
        data = []
    except json.JSONDecodeError:
        # If the file is empty or not valid JSON, start with an empty list
        data = []
    
    # Append the new data
    data.append(new_data)
    
    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def create_dataset_from_responses(input_file, output_file='regen.json'):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    # pool starts out as the seed instructions
    current_pool_instructions = load_seed_instructions()
    num_seed_instructions = len(current_pool_instructions)
    current_pool_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in current_pool_instructions]
    machine_instruction_data = []
    
    print(f"Using {multiprocessing.cpu_count()} CPU threads for major bottleneck of ROGUE-L")

    for message, metadata, finish_reason in response_iterator(input_file):
        instruction_data = post_process_response(len(metadata['seed_idxs']), message, finish_reason)
        
        process_start = time.time()
        total = len(instruction_data)
        keep = 0
        kept_instructions = []
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    current_pool_instructions,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                current_pool_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }

            if max(rouge_scores) > 0.7:
                continue
            
            keep += 1
            machine_instruction_data.append(instruction_data_entry)
            # add generated instruction to pool of all instructions for rouge filtering 
            current_pool_instructions.append(instruction_data_entry["instruction"])
            current_pool_instruction_tokens.append(new_instruction_tokens)

        
        process_duration = time.time() - process_start
        print(f"Processing request idx {metadata['request_idx']} took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        with open(output_file, "w") as f:
            json.dump(machine_instruction_data, f)


def main(task, **kwargs):
    globals()[task](**kwargs)

# Example usage
# python -m read_requests create_dataset_from_responses --input_file=requests_to_parallel_process_results.jsonl
if __name__ == "__main__":
    fire.Fire(main)