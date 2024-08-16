import os
import torch
import torch.distributed as dist

def run():
   local_rank = int(os.environ["LOCAL_RANK"])
   rank = int(os.environ["RANK"])
   world_size = int(os.environ["WORLD_SIZE"])

   dist.init_process_group("nccl")
   print(f"Rank {rank}/{world_size} initialized. Local rank: {local_rank}")

   if torch.cuda.is_available():
       device = torch.device(f"cuda:{local_rank}")
       torch.cuda.set_device(device)
       print(f"Rank {rank} using GPU: {torch.cuda.get_device_name(device)}")
   else:
       print(f"Rank {rank} CUDA not available, using CPU")

   tensor = torch.ones(1)
   if torch.cuda.is_available():
       tensor = tensor.to(device)
   dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
   print(f"Rank {rank} All-reduce sum: {tensor.item()}")

   dist.destroy_process_group()

if __name__ == "__main__":
    run()
