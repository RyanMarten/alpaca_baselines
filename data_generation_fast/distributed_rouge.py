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

def launch_slurm_job(num_nodes: int = 64):
    slurm_script = f"""#!/bin/bash
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --job-name=rouge_filter
#SBATCH --output=rouge_filter_%j.out

srun python -m create_dataset filter_instructions_rouge_distributed
"""
    with open("slurm_job.sh", "w") as f:
        f.write(slurm_script)
    
    subprocess.run(["sbatch", "slurm_job.sh"])

def filter_instructions_rouge_distributed(input_file: str, output_file: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Master process
        with open(input_file, "r") as f:
            instructions = json.load(f)

        # Create shards
        shards = np.array_split(instructions, size - 1)
        for i, shard in enumerate(shards):
            with open(f"shard_{i}.json", "w") as f:
                json.dump(shard.tolist(), f)

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        filtered = []
        not_filtered = []
        current_pool_instructions = []

        for idx, item in enumerate(instructions):
            inst = item["instruction"]
            input_text = item["input"]
            output_text = item["output"]

            process_start = time.time()

            new_instruction_tokens = scorer._tokenizer.tokenize(inst)
            
            # Send instruction to all worker processes
            for i in range(1, size):
                comm.send((inst, new_instruction_tokens), dest=i)

            # Collect results from worker processes
            max_scores = []
            max_instructions = []
            for i in range(1, size):
                result = comm.recv(source=i)
                max_scores.append(result[0])
                max_instructions.append(result[1])

            # Process results
            if max(max_scores) >= 0.7:
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
                current_pool_instructions.append(inst)
                shard_to_update = idx % (size - 1)
                comm.send(("update", inst, new_instruction_tokens), dest=shard_to_update + 1)

            process_duration = time.time() - process_start
            print(f"Processing example {idx} took {process_duration:.2f}s")

        # Signal worker processes to finish
        for i in range(1, size):
            comm.send(("finish", None, None), dest=i)

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

        while True:
            data = comm.recv(source=0)
            if data[0] == "finish":
                break
            elif data[0] == "update":
                update_shard(shard_file, data[1], data[2])
            else:
                instruction, instruction_tokens = data
                result = compute_rouge_scores(shard_file, instruction, instruction_tokens, scorer)
                comm.send(result, dest=0)

def compute_rouge_scores(shard_file: str, instruction: str, instruction_tokens: List[str], scorer: rouge_scorer.RougeScorer) -> Tuple[float, str]:
    if not os.path.exists(shard_file):
        return 0, ""

    with open(shard_file, "r") as f:
        shard_data = json.load(f)
    
    max_score = 0
    max_instruction = ""

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        scores = p.map(
            partial(rouge_scorer._score_lcs, instruction_tokens),
            [scorer._tokenizer.tokenize(item["instruction"]) for item in shard_data]
        )

    for i, score in enumerate(scores):
        if score.fmeasure > max_score:
            max_score = score.fmeasure
            max_instruction = shard_data[i]["instruction"]

    return max_score, max_instruction

def update_shard(shard_file: str, instruction: str, instruction_tokens: List[str]):
    with open(shard_file, "r") as f:
        shard_data = json.load(f)
    
    shard_data.append({"instruction": instruction, "tokens": instruction_tokens})
    
    with open(shard_file, "w") as f:
        json.dump(shard_data, f)

def print_stats(filtered: List[Dict], not_filtered: List[Dict]):
    print("\nFiltered instruction counts:")
    print(f"  ROUGE-L similarity: {len(filtered)}")
    total_filtered = len(filtered)
    total_count = len(filtered) + len(not_filtered)
    print(f"Total filtered: {total_filtered} out of {total_count}")
    percentage_filtered = (total_filtered / total_count) * 100 if total_count > 0 else 0
    print(f"That's {percentage_filtered:.2f}% filtered out of total")
    print(f"Kept {len(not_filtered)} instructions")