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
import logging

# Configure logging
def get_logger():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    node_name = os.environ.get('SLURMD_NODENAME', 'Unknown')
    process_type = "MASTER" if rank == 0 else "WORKER"
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter(f'%(asctime)s - {process_type} - {node_name} - %(levelname)s - %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Disable propagation to avoid double logging
    logger.propagate = False
    return logger

logger = get_logger()

TERMINATION_MSG = "TERMINATE"
ADD_TO_SHARD_MSG = "ADD_TO_SHARD"
CALCULATE_ROUGE_MSG = "CALCULATE_ROUGE"

def filter_rouge(input_file: str, output_file: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Master process
        logger.info(f"Master process (rank {rank}) started. Total processes: {size}")

        # Initialize shards with seed tasks instructions
        intialize_start = time.time()
        seed_tasks = [json.loads(l) for l in open("../seed_tasks.jsonl", "r")]
        seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks]
        logger.info(f"Loaded {len(seed_instruction_data)} human-written seed instructions")
        
        for idx, task in enumerate(seed_instruction_data):
            inst = task['instruction']
            shard_idx = idx % (size - 1) + 1
            comm.send((ADD_TO_SHARD_MSG, inst), dest=shard_idx)
        logger.info(f"Sent seed instructions to shards")

        intialize_duration = time.time() - intialize_start
        logger.info(f"Initializing shards took {intialize_duration:.2f}s")

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
                comm.send((CALCULATE_ROUGE_MSG, inst), dest=i)
                logger.debug(f"Master sent '{inst}' to process {i}")

            # Collect results from worker processes
            max_scores = []
            max_instructions = []
            for i in range(1, size):
                result = comm.recv(source=i)
                max_scores.append(result[0])
                max_instructions.append(result[1])
                logger.debug(f"Master received: {result} from process {i}")

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
                shard_idx = (idx % (size - 1)) + 1
                comm.send((ADD_TO_SHARD_MSG, inst), dest=shard_idx)
                
            process_duration = time.time() - process_start
            logger.debug(f"Calculating all rouge for example {idx} took {process_duration:.2f}s")
            
        # Signal worker processes to finish
        for i in range(1, size):
            comm.send((TERMINATION_MSG, None), dest=i)
            logger.debug(f"Master sent termination message to process {i}")

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


        # Calculate and log total time
        total_duration = time.time() - intialize_start
        logger.info(f"Total processing time: {total_duration:.2f}s")
        
        # Log additional statistics
        logger.info(f"Total instructions processed: {len(filtered) + len(not_filtered)}")
        logger.info(f"Instructions filtered (ROUGE-L): {len(filtered)}")
        logger.info(f"Instructions kept: {len(not_filtered)}")
        logger.info(f"Filtering rate: {len(filtered) / (len(filtered) + len(not_filtered)) * 100:.2f}%")
        logger.info(f"Average processing time per instruction: {total_duration / (len(filtered) + len(not_filtered)):.4f}s")

    else:
        # Worker processes
        logger.info(f"I have {multiprocessing.cpu_count()} cores")

        shard_id = rank - 1
        shard_file = f"{os.environ.get('SCRATCH', '.')}/rougue_shards/shard_{shard_id}.json"
        # Remove shard file if it exists
        if os.path.exists(shard_file):
            os.remove(shard_file)
            logger.debug(f"Process {rank} removed existing shard file: {shard_file}")
        
        # Create new shard file
        os.makedirs(os.path.dirname(shard_file), exist_ok=True)
        logger.debug(f"Created new shard file: {shard_file}")
        with open(shard_file, "w") as f:
            json.dump([], f)

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        last_modified = 0
        shard_instructions = []
        shard_instruction_tokens = []

        while True:
            message = comm.recv(source=0)
            if message[0] == TERMINATION_MSG:
                logger.debug(f"Process {rank} received termination message. Exiting.")
                break
            elif message[0] == ADD_TO_SHARD_MSG:
                inst = message[1]
                start_time = time.time()
                add_to_shard(inst, shard_file)
                shard_instructions.append(inst)
                shard_instruction_tokens.append(scorer._tokenizer.tokenize(inst))
                duration = time.time() - start_time
                logger.debug(f"Process {rank} added '{inst}' to shard in {duration:.4f} seconds")
            elif message[0] == CALCULATE_ROUGE_MSG:
                inst = message[1]
                logger.debug(f"Process {rank} received: {inst}")
                
                if os.path.exists(shard_file):
                    # Check if shard file has been modified since last read
                    if os.path.getmtime(shard_file) > last_modified:
                        last_modified = os.path.getmtime(shard_file)
                        # reload file and compute tokens for it
                        update_start = time.time()
                        with open(shard_file, "r") as f:
                            shard_instructions = json.load(f)
                        shard_instruction_tokens = [scorer._tokenizer.tokenize(shard_inst) for shard_inst in shard_instructions]
                        update_duration = time.time() - update_start
                        logger.debug(f"Process {rank} updated tokenizer in {update_duration:.4f} seconds")
                    # then compute rogue scores
                    start_time = time.time()
                    result = compute_rouge_scores(inst, shard_instructions, shard_instruction_tokens, scorer)
                    duration = time.time() - start_time
                    logger.debug(f"Process {rank} calculated ROUGE for example in {duration:.4f} seconds")
                else: 
                    result = (0, "")
                    logger.warning(f"Process {rank} sent {result} to master (shard file not found)")
                

                comm.send(result, dest=0)
                logger.debug(f"Process {rank} sent {result} to master")

def add_to_shard(inst: str, shard_file: str):
    # Efficient I/O rewrite of the file
    with open(shard_file, "r+") as f:
        try:
            shard_instructions = json.load(f)
        except json.JSONDecodeError:
            shard_instructions = []
        shard_instructions.append(inst)
        f.seek(0)
        json.dump(shard_instructions, f)
        f.truncate()

def compute_rouge_scores(instruction: str, shard_instructions: List[str], shard_instruction_tokens: List[List[str]], scorer: rouge_scorer.RougeScorer) -> Tuple[float, str]:    
    instruction_tokens = scorer._tokenizer.tokenize(instruction)

    # Use numpy for faster array operations
    scores = np.array([rouge_scorer._score_lcs(instruction_tokens, tokens).fmeasure for tokens in shard_instruction_tokens])
    
    # Find the index of the maximum score
    max_index = np.argmax(scores)
    
    return scores[max_index], shard_instructions[max_index]

def main(task, **kwargs):
    globals()[task](**kwargs)

# Example usage
# python -m distributed_rouge filter_rouge --input_file test_regen.json --output_file filtered.json
if __name__ == "__main__":
    fire.Fire(main)
