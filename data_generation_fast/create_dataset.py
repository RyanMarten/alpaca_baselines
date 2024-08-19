import json
import os
import pdb
import re 
import string
from rouge_score import rouge_scorer
from tqdm import tqdm
import multiprocessing
from functools import partial
import numpy as np
import time
import json
import fire

def instruction_iterator(instructions_filepath):
    with open(instructions_filepath, 'r') as file_in:
        instructions = json.loads(file_in.read())
        for instruction_data in instructions:
            try:
                inst = instruction_data['instruction']
                input = instruction_data['input']
                output = instruction_data['output']
                yield inst, input, output
            except Exception as e:
                print(f"Error processing instruction: {e}")
                continue

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

def parse_single_response(num_prompt_instructions, message, finish_reason):
    # original used completions API, I'm using chat completions
    # the chat model repeates the 4. Instruction: in it's response already *about 2/3 of the time*
    # so only add the prefix if it isn't there already
    raw_instructions = message
    prefix = f"{num_prompt_instructions+1}. Instruction: "
    if raw_instructions[:len(prefix)] != prefix:
        raw_instructions = prefix + message
        
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    failed_to_parse = []
    truncated = []

    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and finish_reason == "length":
            truncated.append(inst)
            continue
       
        # parse the string to get instruction, input, and output
        """Example of expected instruction with 7 fields
        ['\n', 
        'Instruction', 
        ' Rewrite this sentence in the passive voice.\n',
        'Input',
        '\nThe chef prepared a stunning feast for the guests.\n',
        'Output',
        '\nA stunning feast was prepared for the guests by the chef.\n']
        """
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            if inst.strip() != '': # the original regex can create whitespace splits
                failed_to_parse.append(inst)
            continue
    
        # succcessfully parsed string, convert into json dict
        inst = splitted_data[2].strip()
        input = splitted_data[4].strip()
        input = "" if input.lower() == "<noinput>" else input
        output = splitted_data[6].strip()
        instructions.append({"instruction": inst, "input": input, "output": output})

    return instructions, truncated, failed_to_parse

def parse_all_responses(input_file, output_file_parsed, output_file_failed):
    instructions, truncated, failed_to_parse = [],[],[]

    for message, metadata, finish_reason in response_iterator(input_file):
        insts, truncs, fails = parse_single_response(len(metadata['seed_idxs']), message, finish_reason)
        instructions.extend(insts)
        truncated.extend(truncs)
        failed_to_parse.extend(fails)

    print(f"Successfully parsed {len(instructions)} instructions")
    print(f"{len(truncated)+len(failed_to_parse)} failed to parse")
    print(f"{len(truncated)} which were due to response trunctation")

    with open(output_file_parsed, "w") as f:
        json.dump(instructions, f)

    with open(output_file_failed, "w") as f:
        json.dump(truncated + failed_to_parse, f)

def filter_instructions_heuristics(input_file, output_file):
    too_short = []
    too_long = []
    blacklisted = []
    write_a_program_prefix = []
    punctuation_prefix = []
    ascii_prefix = []
    not_filtered = []
    total_count = 0

    for inst, input_text, output_text in instruction_iterator(input_file):    
        total_count += 1

        # filter out too short or too long instructions
        if len(inst.split()) <= 3:
            too_short.append({"instruction": inst, "input": input_text, "output": output_text, "filtered_reason": "too_short"})
            continue

        if len(inst.split()) > 150:
            too_long.append({"instruction": inst, "input": input_text, "output": output_text, "filtered_reason": "too_long"})
            continue
        
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image", "images", "graph", "graphs", "picture", "pictures",
            "file", "files", "map", "maps", "draw", "plot", "go to",
            "video", "audio", "music", "flowchart", "diagram",
        ]
        if any(find_word_in_string(word, inst) for word in blacklist):
            blacklisted.append({"instruction": inst, "input": input_text, "output": output_text, "filtered_reason": "blacklisted"})
            continue
        
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit confusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            write_a_program_prefix.append({"instruction": inst, "input": input_text, "output": output_text, "filtered_reason": "write_a_program_prefix"})
            continue
        
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            punctuation_prefix.append({"instruction": inst, "input": input_text, "output": output_text, "filtered_reason": "punctuation_prefix"})
            continue
        
        # filter those starting with non-english character
        if not inst[0].isascii():
            ascii_prefix.append({"instruction": inst, "input": input_text, "output": output_text, "filtered_reason": "ascii_prefix"})
            continue

        # If not filtered, add to not_filtered list
        not_filtered.append({"instruction": inst, "input": input_text, "output": output_text})

    # Write not filtered instructions back to input file
    with open(input_file, "w") as f:
        json.dump(not_filtered, f)

    # Write filtered instructions to output file
    filtered = too_short + too_long + blacklisted + write_a_program_prefix + punctuation_prefix + ascii_prefix
    with open(output_file, "w") as f:
        json.dump(filtered, f)

    # Print the counts in a formatted way
    filtered_counts = {
        "Too short": len(too_short),
        "Too long": len(too_long),
        "Blacklisted": len(blacklisted),
        "Write a program prefix": len(write_a_program_prefix),
        "Punctuation prefix": len(punctuation_prefix),
        "Non-ASCII prefix": len(ascii_prefix)
    }
    print("\nFiltered instruction counts:")
    for category, count in filtered_counts.items():
        print(f"  {category}: {count}")
    total_filtered = sum(filtered_counts.values())
    print(f"Total filtered: {total_filtered} out of {total_count}")
    # Calculate the percentage of filtered instructions
    percentage_filtered = (total_filtered / total_count) * 100 if total_count > 0 else 0
    print(f"That's {percentage_filtered:.2f}% filtered out of total")

def filter_instructions_rouge(input_file, output_file):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    # pool starts out as the seed instructions
    current_pool_instructions = load_seed_instructions()
    num_seed_instructions = len(current_pool_instructions)
    current_pool_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in current_pool_instructions]
    not_filtered = []
    filtered = []
    
    print(f"Using {multiprocessing.cpu_count()} CPU threads for major bottleneck of ROGUE-L")

    total_instructions = sum(1 for _ in instruction_iterator(input_file))
    for inst, input_text, output_text in tqdm(instruction_iterator(input_file), total=total_instructions, desc="Processing instructions"):
        # process_start = time.time()

        # computing similarity with the pre-tokenzied instructions
        new_instruction_tokens = scorer._tokenizer.tokenize(inst)

        # Use numpy for faster array operations
        rouge_scores = np.array([rouge_scorer._score_lcs(new_instruction_tokens, tokens).fmeasure for tokens in current_pool_instruction_tokens])

        if max(rouge_scores) > 0.7:
            filtered.append({
                "instruction": inst,
                "input": input_text,
                "output": output_text,
                "filtered_reason": "rouge_similarity",
                "most_similar_score": max(rouge_scores),
                "most_similar_instruction": current_pool_instructions[np.argmax(rouge_scores)]
            })
            continue

        # If not filtered, add to not_filtered list
        not_filtered.append({"instruction": inst, "input": input_text, "output": output_text})
        # add generated instruction to pool of all instructions for rouge filtering 
        current_pool_instructions.append(inst)
        current_pool_instruction_tokens.append(new_instruction_tokens)

        # process_duration = time.time() - process_start
        # print(f"Processing this example took {process_duration:.2f}s")

    # Write not filtered instructions back to input file
    with open(input_file, "w") as f:
        json.dump(not_filtered, f)

    # Load existing filtered instructions if the file exists, otherwise start with an empty list
    all_filtered = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            all_filtered = json.load(f)
    # Append new filtered instructions to the existing ones
    all_filtered.extend(filtered)
    # Write updated filtered instructions back to output file
    with open(output_file, "w") as f:
        json.dump(all_filtered, f)

    # Print the counts in a formatted way
    print("\nFiltered instruction counts:")
    print(f"  ROUGE-L similarity: {len(filtered)}")
    total_filtered = len(filtered)
    total_count = len(filtered) + len(not_filtered)
    print(f"Total filtered: {total_filtered} out of {total_count}")
    # Calculate the percentage of filtered instructions
    percentage_filtered = (total_filtered / total_count) * 100 if total_count > 0 else 0
    print(f"That's {percentage_filtered:.2f}% filtered out of total")
    print(f"Kept {len(not_filtered)} instructions")

def main(task, **kwargs):
    globals()[task](**kwargs)

# Example usage
# python -m create_dataset parse_all_responses --input_file test_results.jsonl --output_file_parsed test_parsed.json --output_file_failed test_parsing_failed.json
# python -m create_dataset filter_instructions_heuristics --input_file regen.json --output_file filtered.json 
# python -m create_dataset filter_instructions_rouge --input_file test_regen.json --output_file filtered.json 
if __name__ == "__main__":
    fire.Fire(main)