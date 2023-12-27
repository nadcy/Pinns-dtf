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

from user_fun.io import read_comsol
from scipy.interpolate import griddata

from user_fun.geom import line_sample
from user_fun.bc import data_loss_factory
from user_fun.pde import ico_2D_static_factory
from user_fun.field import D2Field

debug_print_flag = True
save_dir = './timepde/multi_gpu/'
from prework import Trainer
def interpolate_puv(file_path, x1, x2, num_points):
    # 读取数据并提取相应的列
    tb = read_comsol.comsol_read(file_path)
    plot_x = tb.values[:,0]
    plot_y = tb.values[:,1]
    plot_p = tb.values[:,2]
    plot_u = tb.values[:,3]
    plot_v = tb.values[:,4]

    # 将x和y坐标组合成输入
    plot_input = np.column_stack((plot_x, plot_y))

    # 生成给定x坐标的点
    x1_coords = np.full(num_points, x1)
    x2_coords = np.full(num_points, x2)
    y_coords = np.linspace(0, 1, num_points)

    coords_x1 = np.column_stack((x1_coords, y_coords))
    coords_x2 = np.column_stack((x2_coords, y_coords))

    # 使用插值方法获取点云数据
    unique_plot_input, idx = np.unique(plot_input, axis=0, return_index=True)
    p1 = griddata(unique_plot_input, plot_p[idx], coords_x1, method='cubic')
    p2 = griddata(unique_plot_input, plot_p[idx], coords_x2, method='cubic')
    u1 = griddata(unique_plot_input, plot_u[idx], coords_x1, method='cubic')
    u2 = griddata(unique_plot_input, plot_u[idx], coords_x2, method='cubic')
    v1 = griddata(unique_plot_input, plot_v[idx], coords_x1, method='cubic')
    v2 = griddata(unique_plot_input, plot_v[idx], coords_x2, method='cubic')

    # 结合点云数据
    result_y1 = np.column_stack((p1, u1, v1))
    result_y2 = np.column_stack((p2, u2, v2))

    return coords_x1, result_y1, coords_x2, result_y2


def get_problem(seg_density=32):
    file_path = "./timepde/recons/half_cl.csv"
    x1 = 0.1
    x2 = 2.9
    num_points = 10
    result_x1, result_y1, result_x2, result_y2 = interpolate_puv(file_path, x1, x2, num_points)
    field = D2Field([0,1],[0,3])
    pde_input = field.get_field_mesh([seg_density,seg_density*3])
    center = np.array([0.0, 1.3])
    threshold_distance = 0.3
    distances = np.linalg.norm(pde_input - center, axis=1)
    pde_input = pde_input[distances > threshold_distance]

    # 保留wall相关的部分
    bc_left1_x = line_sample([0,0],[0,1], seg_density*2)
    bc_left2_x = line_sample([0,1.6],[0,3], seg_density*2)
    bc_right_x = line_sample([1,3],[1,0], seg_density*3)

    def generate_points_on_semicircle(n, center_x, center_y, radius):
        angles1 = np.linspace(0, 0.5*np.pi, n//2)
        angles2 = np.linspace(1.5*np.pi, 2*np.pi, n//2)
        angles = np.concatenate([angles1, angles2], axis = 0)
        x_coords = center_x + radius * np.cos(angles)
        y_coords = center_y + radius * np.sin(angles)

        points = np.column_stack((x_coords, y_coords))
        return points

    bc_left_circle = generate_points_on_semicircle(64, 0, 1.3, 0.3)

    wall_input = np.vstack([bc_left1_x, bc_left2_x, bc_right_x, bc_left_circle])
    wall_output = np.zeros([wall_input.shape[0], 2])

    # 使用result_x1, result_y1, result_x2, result_y2替换inlet和outlet部分
    # 交换x, y分量与u, v分量
    data_input = np.vstack((result_x1[:, [1,0]], result_x2[:, [1,0]]))
    data_output = np.vstack((result_y1[:, [0,2,1]], result_y2[:, [0,2,1]]))

    pde_output = np.zeros([pde_input.shape[0], 3])

    loss_fn = torch.nn.MSELoss()
    loss_list = [
        ico_2D_static_factory(loss_fn, 0.01),
        data_loss_factory(loss_fn, [0, 1, 2]),  # Data loss for x1 and x2 (p, u, v)
        data_loss_factory(loss_fn, [1, 2]),  # wall (u, v) == 0
    ]

    cloud_point_list = [
        [pde_input, pde_output],
        [data_input, data_output],
        [wall_input, wall_output],
    ]
    return cloud_point_list, loss_list


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
    tr1 = Trainer(rank,size,[[0,1.1],[1,2.1],[2,3]],'udPINNs',True, device,
                 send_graph = [[1],[0,2],[1]],
                 send_t_span_list = [[[1,1.1]],[[1,1.1],[2,2.1]],[[2,2.1]]],
                 comm_weight = [[1],[0.1,0.1],[1]])
    tr1.assign_task(cloud_point_list=cloud_point_list,
                   loss_list=loss_list, output_dim=3)
    tr1.train(10000,10,10,'para_result/flow-step1',model_vec=[2,32,32,32,3])


    tr2 = Trainer(rank,size,[[0,1.1],[1,2.1],[2,3]],'udPINNs',True, device,
                 send_graph = [[1],[0,2],[1]],
                 send_t_span_list = [[[1,1.1]],[[1,1.1],[2,2.1]],[[2,2.1]]],
                 comm_weight = [[1],[1,1],[1]])
    tr2.assign_task(cloud_point_list=cloud_point_list,
                   loss_list=loss_list, output_dim=3)
    tr2.train(1000,10,10,'para_result/flow-step2',model_vec=tr1.solver._model)
    
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