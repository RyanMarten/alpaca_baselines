from mpi4py import MPI
import socket

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = socket.gethostname()

    if rank == 0:
        # Master process
        print(f"Master on {hostname} (rank {rank}) started. Total processes: {size}")
        for i in range(1, size):
            data = f"Hello, process {i}!"
            comm.send(data, dest=i)
            print(f"Master on {hostname} sent message to process {i}")
        
        for i in range(1, size):
            response = comm.recv(source=i)
            print(f"Master on {hostname} received: {response}")

    else:
        # Worker processes
        data = comm.recv(source=0)
        print(f"Process {rank} on {hostname} received: {data}")
        response = f"Greetings from process {rank} on {hostname}!"
        comm.send(response, dest=0)
        print(f"Process {rank} on {hostname} sent response to master")

if __name__ == "__main__":
    main()