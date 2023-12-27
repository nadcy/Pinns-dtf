"""run.py:"""
#!/usr/bin/env python
import time
import math
import os
import sys

sys.path.append('./')
print(sys.path)

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
np.set_printoptions(threshold=8)
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

np.random.seed(30)
torch.manual_seed(30)

from user_fun.baseline.wave import WaveBenchMark

debug_print_flag = True
save_dir = './timepde/multi_gpu/'
from prework import Trainer




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
    cloud_point_list,loss_list = WaveBenchMark()
    tr1 = Trainer(rank,size,[[0,1.1],[1,2.1],[2,3]],'XPINNs',True, device,
                 send_graph = [[1],[0,2],[1]],
                 send_t_span_list = [[[1,1.1]],[[1,1.1],[2,2.1]],[[2,2.1]]],
                 comm_weight = [[1],[1,1],[1]])
    tr1.assign_task(cloud_point_list=cloud_point_list,
                   loss_list=loss_list, output_dim=1)
    tr1.train(2000,10,10,'para_result/wave-x',model_vec=[2,64,64,32,1])

    
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