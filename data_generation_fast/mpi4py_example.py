from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Master process
        print(f"Master process (rank {rank}) started. Total processes: {size}")
        for i in range(1, size):
            data = f"Hello, process {i}!"
            comm.send(data, dest=i)
            print(f"Master sent message to process {i}")
        
        for i in range(1, size):
            response = comm.recv(source=i)
            print(f"Master received: {response}")

    else:
        # Worker processes
        data = comm.recv(source=0)
        print(f"Process {rank} received: {data}")
        response = f"Greetings from process {rank}!"
        comm.send(response, dest=0)
        print(f"Process {rank} sent response to master")

if __name__ == "__main__":
    main()