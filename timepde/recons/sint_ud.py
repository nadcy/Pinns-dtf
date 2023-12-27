"""run.py:"""
#!/usr/bin/env python
import time
import os
import sys
sys.path.append('./')
print(sys.path)

import numpy as np
np.set_printoptions(threshold=8)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from user_fun.baseline import sint

debug_print_flag = True
save_dir = './timepde/multi_gpu/'
from prework import Trainer

t_span = [[0, 2.1],[2,4.1],[4,6]]

def test_eval(x_test_whole,y_test_whole):
    error = np.mean(np.abs(np.cos(np.pi*x_test_whole) - y_test_whole))
    return error

def run(rank, size):

    print(f'rank:{rank},size:{size}:ready')
    loss_fn = torch.nn.MSELoss()

    if torch.cuda.device_count()>=3:
        print('use_multi_gpu')
        device = \
            torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    else:
        print('use_single_gpu')
        device = \
            torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    # 原始问题定义
    cloud_point_list,loss_list = sint.get_long_time_problem(pde_epoch_size=128)
    tr = Trainer(rank,size,t_span,'udPINNs',True, device,
                 eval_func = test_eval)

    tr.assign_task(cloud_point_list=cloud_point_list, loss_list=loss_list, output_dim=1)
    tr.train(600,10,10,'para_result/sint-ud')
    end_time = time.time()

    time_delta = end_time - start_time
    print(f'time delta is {time_delta}')

    

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = len(t_span)
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print('-------')