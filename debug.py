import sys
sys.path.append('../')

import numpy as np
import torch
import matplotlib

from user_fun.get_net import ForwardNetwork
from user_fun.solver.cp_solver import CloudPointSolver
from user_fun.baseline import sint
from user_fun.bc import data_loss_factory

debug_print_flag = True
# 保证迭代参数与DEEPXDE一致
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ForwardNetwork([1, 50, 50, 50, 1]).to(device)
loss_fn = torch.nn.MSELoss()

cloud_point_list,loss_list = sint.get_long_time_problem()

t_span_list = np.array([[0,2.2],[2,4.2],[4,6]])
model_num = len(t_span_list)
send_t_span = [None for i in range(len(t_span_list))]
for i in range(len(t_span_list)):
    if i == len(t_span_list)-1:
        send_t_span[i] = []
    else:
        send_t_span[i] = [t_span_list[i+1][0],t_span_list[i][1]]

if debug_print_flag:
    print('send_t_span', send_t_span)


""" 对数据集进行切分，默认x_data最后一维是时间维，x_data中涉及到的在交界区域内的任意类
    型的采样点均会被视为交界损失的发生点。
"""
def get_span_cp_loss(cloud_point_list,loss_list,t_span):
    """
    len(cloud_point_list) : 损失项的数量
    """
    res_cloud_point_list = []
    res_loss_list = []
    if not(t_span is None):
        for cp_item,loss_item in zip(cloud_point_list,loss_list):
            x_data,y_data = cp_item
            idx = (x_data[:,-1]>=t_span[0]) & (x_data[:,-1]<=t_span[1])
            choose_x_data = x_data[idx,:]
            choose_y_data = y_data[idx,:]
            if len(choose_x_data)!=0:
                res_cloud_point_list.append([choose_x_data, choose_y_data])
                res_loss_list.append(loss_item)
        return res_cloud_point_list,res_loss_list

# 初值赋予
cp_para_list = []
loss_para_list = []
solver_list = []
for model_id in range(model_num):
    cp_para_item, loss_para_item = \
        get_span_cp_loss(cloud_point_list, loss_list, t_span_list[model_id])
    cp_para_list.append(cp_para_item)
    loss_para_list.append(loss_para_item)
    solver_list.append(CloudPointSolver(
        model = [1, 100, 100, 100, 1],
        optimizer = "adam"))

epoch_num = 500
info_exchange_round = 10
test_round = 30

# 初始化通信组件
send_xdata_list = [None for i in range(model_num)]
send_cp_list = [None for i in range(model_num)]
recv_cp_list = [None for i in range(model_num)]
for model_id in range(model_num):
    if model_id != model_num - 1:
        cp_para_item, _ = \
            get_span_cp_loss(cloud_point_list, loss_list, send_t_span[model_id])
        x_data_item_list = [item[0] for item in cp_para_item]
        send_xdata_list[model_id] = np.vstack(x_data_item_list)

data_loss = data_loss_factory(loss_fn)

for epoch in range(epoch_num):
    if epoch % info_exchange_round == 0:
        # 采样
        for model_id in range(model_num):
            if model_id!=model_num-1:
                y_data = \
                    solver_list[model_id].model_eval(send_xdata_list[model_id])
                send_cp_list[model_id] = (send_xdata_list[model_id], y_data)
        # 通信
        for model_id in range(model_num):     
            if model_id != 0:
                recv_cp_list[model_id] = send_cp_list[model_id-1]

    if epoch % test_round == 0:
        solver_list[model_id].test_step(
            cloud_point_list = \
                cp_para_list[model_id] + [recv_cp_list[model_id]],
            loss_list = loss_para_list[model_id] + [data_loss],
            print_flag = True
        )

    solver_list[model_id].train_step(
        cloud_point_list = \
            cp_para_list[model_id] + [recv_cp_list[model_id]], 
        loss_list = loss_para_list[model_id] + [data_loss]
    )

