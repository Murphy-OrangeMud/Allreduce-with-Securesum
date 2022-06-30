import torch
import torch.distributed as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def allreduce_chunk(send, recv):
    rank = dist.get_rank()
    size = dist.get_world_size()
    flag = False

    dist.barrier()

    send_buff = send.clone()
    recv_buff = recv.clone()

    send_chunk = list(torch.chunk(send_buff, size))
    recv_chunk = list(torch.chunk(recv_buff, size))
    if len(send_chunk) < size:
        send_chunk.append(torch.ones(send_chunk[0].shape).to(device))
        recv_chunk.append(torch.ones(recv_chunk[0].shape).to(device))
        flag = True

    noise = torch.rand(send_chunk[rank].shape).to(device)
    send_chunk[rank][:] += noise[:]

    left = ((rank - 1) + size) % size
    right = (rank + 1) % size

    # scatter-reduce
    for i in range(size - 1):
        send_req = dist.isend(send_chunk[(rank - i + size) % size], right)
        dist.recv(recv_chunk[(left - i + size) % size], left)
        send_chunk[(left - i + size) % size][:] += recv_chunk[(left - i + size) % size][:]

        send_req.wait()

        dist.barrier()

    send_req = dist.isend(send_chunk[(rank + 1) % size], right)
    dist.recv(send_chunk[(left + 1) % size], left)

    send_req.wait()

    dist.barrier()

    send_chunk[rank][:] -= noise[:]

    # all gather reduce
    for i in range(size - 1):
        send_req = dist.isend(send_chunk[(rank - i + size) % size], right)
        dist.recv(send_chunk[(left - i + size) % size], left)

        send_req.wait()

        dist.barrier()

    if flag:
        send_chunk = send_chunk[:-1]

    recv[:] = torch.concat(send_chunk)

