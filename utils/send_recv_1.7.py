import torch
import torch.distributed as dist
import pickle
import io

def send_object(obj, dst, tag, group=dist.group.WORLD):
    buffer = io.BytesIO()
    pickle.dump(obj, buffer)
    buffer.seek(0)
    serialized_data = buffer.read()

    length_tensor = torch.tensor(len(serialized_data), dtype=torch.int64)
    dist.send(length_tensor, dst, group=group, tag=tag)

    data_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(serialized_data))
    dist.send(data_tensor, dst, group=group, tag=tag+1)

def recv_object(src, tag, group=dist.group.WORLD):
    length_tensor = torch.tensor(0, dtype=torch.int64)
    dist.recv(length_tensor, src, group=group, tag=tag)
    length = length_tensor.item()

    data_tensor = torch.ByteTensor(length)
    dist.recv(data_tensor, src, group=group, tag=tag+1)

    buffer = io.BytesIO(data_tensor.numpy().tobytes())
    obj = pickle.load(buffer)
    return obj

