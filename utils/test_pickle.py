import io
import os
import pickle

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

def send_object(obj, dst, tag=0, group=dist.group.WORLD):
    buffer = io.BytesIO()
    pickle.dump(obj, buffer)
    buffer.seek(0)
    serialized_data = buffer.read()

    length_tensor = torch.tensor(len(serialized_data), dtype=torch.int64)
    dist.send(length_tensor, dst, group=group, tag=tag)

    data_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(serialized_data))
    dist.send(data_tensor, dst, group=group, tag=tag+1)

def recv_object(src, tag=0, group=dist.group.WORLD):
    length_tensor = torch.tensor(0, dtype=torch.int64)
    dist.recv(length_tensor, src, group=group, tag=tag)
    length = length_tensor.item()

    data_tensor = torch.ByteTensor(length)
    dist.recv(data_tensor, src, group=group, tag=tag+1)

    buffer = io.BytesIO(data_tensor.numpy().tobytes())
    obj = pickle.load(buffer)
    return obj

def run(rank, size):
    if rank == 0:
        tensor_a = torch.ones(3, 4)
        send_object(tensor_a, dst=1)
        print("Tensor 已发送")

    if rank == 1:
        src_rank = 0
        tensor_b = recv_object(src_rank)
        print("从进程 0 接收到的 Tensor:\n", tensor_b)

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 3
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print('-------')