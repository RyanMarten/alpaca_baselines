import os
import json
import numpy as np
from functools import partial
import multiprocessing
from rouge_score import rouge_scorer
import time
from typing import List, Tuple, Dict
import subprocess
from mpi4py import MPI
import fire

TERMINATION_MSG = "TERMINATE"

def filter_rouge(input_file: str, output_file: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Master process
        print(f"Master process (rank {rank}) started. Total processes: {size}")

        # Initialize shards with seed tasks instructions
        intialize_start = time.time()
        seed_tasks = [json.loads(l) for l in open("../seed_tasks.jsonl", "r")]
        seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks]
        print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
        
        shard_idx = 0
        for idx, task in enumerate(seed_instruction_data):
            inst = task['instruction']
            shard_idx = add_to_shards(inst, shard_idx, size)
        print(f"Write seed instructions to shards")

        intialize_duration = time.time() - intialize_start
        print(f"Initalizing shards took {intialize_duration:.2f}s")

        with open(input_file, "r") as f:
            instructions = json.load(f)

        filtered = []
        not_filtered = []

        for idx, item in enumerate(instructions):
            inst = item["instruction"]
            input_text = item["input"]
            output_text = item["output"]

            process_start = time.time()
            # Send instruction to all worker processes
            for i in range(1, size):
                comm.send((inst, inst), dest=i)
                print(f"Master sent '{inst}' to process {i}")

            # Collect results from worker processes
            max_scores = []
            max_instructions = []
            for i in range(1, size):
                result = comm.recv(source=i)
                max_scores.append(result[0])
                max_instructions.append(result[1])
                print(f"Master received: {result} from process {i}")

            # Process results
            if max(max_scores) > 0.7:
                filtered.append({
                    "instruction": inst,
                    "input": input_text,
                    "output": output_text,
                    "filtered_reason": "rouge_similarity",
                    "scores": max_scores,
                    "similar_instructions": max_instructions
                })
            else:
                not_filtered.append({"instruction": inst, "input": input_text, "output": output_text})
                # Add to pool and update shards
                shard_idx = add_to_shards(inst, shard_idx, size)
                
            process_duration = time.time() - process_start
            print(f"Processing example {idx} took {process_duration:.2f}s")

        # Signal worker processes to finish
        for i in range(1, size):
            comm.send((TERMINATION_MSG, None, None), dest=i)
            print(f"Master sent termination message to process {i}")

        # Write results
        with open(input_file, "w") as f:
            json.dump(not_filtered, f)

        all_filtered = []
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                all_filtered = json.load(f)
        all_filtered.extend(filtered)
        with open(output_file, "w") as f:
            json.dump(all_filtered, f)

        print_stats(filtered, not_filtered)

    else:
        # Worker processes
        shard_id = rank - 1
        shard_file = f"shard_{shard_id}.json"
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        last_modified = 0

        while True:
            inst = comm.recv(source=0)
            if inst == TERMINATION_MSG:
                print(f"Process {rank} received termination message. Exiting.")
                break
            else:
                print(f"Process {rank} received: {inst}")
                if os.path.exists(shard_file):
                    # Check if shard file has been modified since last read
                    if os.path.getmtime(shard_file) > last_modified:
                        last_modified = os.path.getmtime(shard_file)
                        # reload file and compute tokens for it
                        with open(shard_file, "r") as f:
                            shard_instructions = json.load(f)
                        shard_instruction_tokens = [scorer._tokenizer.tokenize(shard_inst) for shard_inst in shard_instructions]
                    # then compute rogue scores
                    result = compute_rouge_scores(shard_file, inst, shard_instruction_tokens, scorer)
                else: 
                    result = (0, "")
                    print(f"Process {rank} sent {result} to master (shard file not found)")

                comm.send(result, dest=0)
                print(f"Process {rank} sent {result} to master")


def add_to_shards(inst, shard_idx, size):
    # Ensure the shards directory exists
    os.makedirs("shards", exist_ok=True)

    # Append instruction to the current shard file
    shard_file = f"shards/shard_{shard_idx}.json"
    
    # Create the file if it doesn't exist
    if not os.path.exists(shard_file):
        with open(shard_file, "w") as f:
            json.dump([], f)

    # efficient I/O rewrite of the file
    with open(shard_file, "r+") as f:
        try:
            shard_instructions = json.load(f)
        except json.JSONDecodeError:
            shard_instructions = []
        shard_instructions.append(inst)
        f.seek(0)
        json.dump(shard_instructions, f)
        f.truncate()

    # Update the last_shard index
    next_shard_idx = (shard_idx + 1) % (size - 1)
    return next_shard_idx

def compute_rouge_scores(shard_file: str, instruction: str, shard_instructions: List[str], shard_instruction_tokens: List[List[str]], scorer: rouge_scorer.RougeScorer) -> Tuple[float, str]:    
    max_score = 0
    max_instruction = ""
    instruction_tokens = scorer._tokenizer.tokenize(instruction)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        scores = p.map(
            partial(rouge_scorer._score_lcs, instruction_tokens), shard_instruction_tokens)

    for i, score in enumerate(scores):
        if score.fmeasure > max_score:
            max_score = score.fmeasure
            max_instruction = shard_instructions[i]

    return max_score, max_instruction

def print_stats(filtered: List[Dict], not_filtered: List[Dict]):
    print("\nFiltered instruction counts:")
    print(f"  ROUGE-L similarity: {len(filtered)}")
    total_filtered = len(filtered)
    total_count = len(filtered) + len(not_filtered)
    print(f"Total filtered: {total_filtered} out of {total_count}")
    percentage_filtered = (total_filtered / total_count) * 100 if total_count > 0 else 0
    print(f"That's {percentage_filtered:.2f}% filtered out of total")
    print(f"Kept {len(not_filtered)} instructions")

def main(task, **kwargs):
    globals()[task](**kwargs)

# Example usage
# python -m distributed_rougue filter_rouge--input_file test_regen.json --output_file filtered.json
if __name__ == "__main__":
    fire.Fire(main)
