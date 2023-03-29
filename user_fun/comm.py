import torch
import torch.distributed as dist

def print_ranked(*args, **kwargs):
    message = "".join(str(arg) for arg in args)
    print(f"Rank {dist.get_rank()} {message}\n")

def scatter_objects(scatter_object_list):
    """
    This function scatters objects from the process with rank 0 to all other
    processes in a distributed setting. Process 0 has a list of tensors as input,
    while other processes have an empty list as input.

    Args:
        scatter_object_list (list): A list of tensors to be scattered from process 0.
                                    For other processes, this parameter should be an empty list.

    Returns:
        local_obj (Any): The received object in the current process after scattering.

    Note:
        Only process 0 should provide a list of tensors as input. All other processes
        should provide an empty list.
    """
    local_obj = [None]
    dist.scatter_object_list(local_obj, scatter_object_list, src=0)
    return local_obj[0]



def gather_objects(local_obj):
    """
    This function gathers objects from all processes in a distributed setting
    and collects them in the process with rank 0.

    Args:
        local_obj (Any): The local object to be gathered from each process.

    Returns:
        gathered_objects (list): A list of gathered objects from all processes, only
                                    available at process 0. For other processes, it returns None.

    Note:
        The gathered_objects list is only available at process 0. All other processes
        will receive a None value.
    """
    current_rank = dist.get_rank()
    world_size = dist.get_world_size()
    gathered_objects = [None] * world_size if current_rank == 0 else None
    dist.gather_object(local_obj, gathered_objects, dst=0)
    return gathered_objects if current_rank == 0 else None

def graph_send_tensor_list(
        send_graph, recv_graph, 
        send_tensor_list,
        recv_tensor_shape):
    """
    This function is used globally to send and receive messages between nodes
    in a graph based on their send and receive topologies.

    Args:
        send_graph (list): A list of sender node IDs, reflecting the send
            topology.
        recv_graph (list): A list of receiver node IDs, reflecting the receive
            topology.
        send_tensor_list (list): A list of tensors representing the messages to 
            be sent.
        recv_tensor_shape (tuple): The expected shape of the tensors in the 
            received message list.

    Returns:
        recv_tensor_list (list): A list of received tensors, ordered according 
        to the receive topology.

    Note:
        The order of tensors in recv_tensor_list is determined by the receive
        topology.
    """
    rank = dist.get_rank()
    req_list = []
    recv_tensor_list = \
        [torch.zeros(item) for item in recv_tensor_shape]
    for send_id_item,send_tensor_item in \
        zip(send_graph[rank],send_tensor_list):
        req_item = dist.isend(send_tensor_item, dst = send_id_item)
        req_list.append(req_item)

    # send
    for i,recv_id_item in enumerate(recv_graph[rank]):
        req_item = dist.irecv(recv_tensor_list[i], src = recv_id_item)
        req_list.append(req_item)

    # recv
    for req_item in req_list:
        req_item.wait()
    return recv_tensor_list