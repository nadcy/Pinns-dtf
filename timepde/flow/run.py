"""run.py:"""
#!/usr/bin/env python
import time
import math
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append('./')
print(sys.path)

import numpy as np
np.set_printoptions(threshold=8)
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

np.random.seed(30)
torch.manual_seed(30)

from user_fun.geom import line_sample
from user_fun.bc import data_loss_factory
from user_fun.pde import ico_2D_static_factory
from user_fun.field import D2Field

debug_print_flag = True
save_dir = './timepde/multi_gpu/'
from prework import Trainer

def get_problem(seg_density = 32):
    field = D2Field([0,3],[0,1])
    pde_input = field.get_field_mesh([seg_density*3,seg_density])
    center = np.array([1.3, 0.0])
    threshold_distance = 0.3
    distances = np.linalg.norm(pde_input - center, axis=1)
    pde_input = pde_input[distances > threshold_distance]

    bc_left_x = line_sample([0,0],[0,1], seg_density)
    bc_up_x = line_sample([0,1],[3,1], seg_density*3)
    bc_right_x = line_sample([3,1],[3,0], seg_density)
    bc_down1_x = line_sample([0,0],[1,0], seg_density)
    bc_down2_x = line_sample([1.6,0],[3,0],math.floor(1.4*seg_density))

    def generate_points_on_semicircle(n, center_x, center_y, radius):
        angles = np.linspace(0, np.pi, n)
        x_coords = center_x + radius * np.cos(angles)
        y_coords = center_y + radius * np.sin(angles)

        points = np.column_stack((x_coords, y_coords))
        return points

    bc_down_circle = generate_points_on_semicircle(64, 1.3, 0, 0.3)

    wall_input = np.vstack([ bc_up_x,
                        bc_down1_x,bc_down2_x,bc_down_circle])
    inlet_input = bc_left_x
    outlet_input = bc_right_x

    pde_output = np.zeros([pde_input.shape[0],3])
    wall_output = np.zeros([wall_input.shape[0],2])
    inlet_output = np.zeros([inlet_input.shape[0],2])
    inlet_output[:,0] = 1.0
    outlet_output = np.zeros([outlet_input.shape[0],1])

    loss_fn = torch.nn.MSELoss()
    loss_list = [
        ico_2D_static_factory(loss_fn, 0.01),
        data_loss_factory(loss_fn, [1,2]), #wall (u,v)==0
        data_loss_factory(loss_fn, [1,2]), #inlet (u,v)==(1,0)
        data_loss_factory(loss_fn, [0]), #output (p)==0
    ]

    cloud_point_list = [
        [pde_input,pde_output],
        [wall_input,wall_output],
        [inlet_input,inlet_output],
        [outlet_input,outlet_output]
    ]
    return cloud_point_list,loss_list


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
    cloud_point_list,loss_list = get_problem()
    tr = Trainer(rank,size,[[0,1.1],[1,2.1],[2,3]],'XPINNs',True, device)
    tr.read_real('timepde/flow/half_cl.csv')
    tr.assign_task(cloud_point_list=cloud_point_list, loss_list=loss_list)
    tr.train(1000,10,10,'para_result/tppinns')
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