from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Master process
        print(f"Master process (rank {rank}) started. Total processes: {size}")
        
        # Send some messages to workers
        for i in range(1, size):
            for j in range(3):  # Send 3 messages to each worker
                data = f"Task {j+1} for worker {i}"
                comm.send(data, dest=i)
                print(f"Master sent '{data}' to process {i}")
        
        # Send termination message to all workers
        for i in range(1, size):
            comm.send("DONE", dest=i)
            print(f"Master sent termination message to process {i}")
        
        # Collect results from workers
        for i in range(1, size):
            for j in range(3):  # Expect 3 responses from each worker
                response = comm.recv(source=i)
                print(f"Master received: {response}")

    else:
        # Worker processes
        while True:
            data = comm.recv(source=0)
            if data == "DONE":
                print(f"Process {rank} received termination message. Exiting.")
                break
            
            print(f"Process {rank} received: {data}")
            response = f"Result of {data} processed by worker {rank}"
            comm.send(response, dest=0)
            print(f"Process {rank} sent response to master")

if __name__ == "__main__":
    main()