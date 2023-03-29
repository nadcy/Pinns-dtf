"""run.py:"""
#!/usr/bin/env python
import time
import os
import sys
sys.path.append('./')
print(sys.path)

import numpy as np
np.set_printoptions(threshold=8)
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from user_fun.geom import get_tspan_cp_loss
from user_fun.solver.cp_solver import CloudPointSolver
from user_fun.baseline import sint
from user_fun.bc import data_loss_factory
from user_fun.comm import \
    print_ranked,scatter_objects,gather_objects,graph_send_tensor_list

debug_print_flag = True
save_dir = './timepde/multi_gpu/'

def run(rank, size):
    print(f'rank:{rank},size:{size}:ready')
    loss_fn = torch.nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 原始问题定义
    cloud_point_list,loss_list = sint.get_long_time_problem()
    
    # 预处理
    # Trainer
    #
    # STEP1: global_init
    #   1-1 DefineOrgProblem: cloud_point_list,loss_list
    #   1-2 UserDefine: send_graph,recv_graph,t_span_list
    #
    # STEP2: master_init
    # STEP3: static_var_scatter
    # STEP4: train

    # split_cloud_point_list
    # input:
    
    model_num = size
    t_span_list = np.array([[0,2.2],[2,4.2],[4,6]])
    send_graph = [[1],[2,0],[1]]
    recv_graph = [[] for i in range(model_num)]
    for u,edge_list in enumerate(send_graph):
        for i,v in enumerate(edge_list):
            recv_graph[v].append(u)

    if rank == 0:
        #STEP1-1 Determine overlapping tpan: t_span_list -> send_t_span_list
        send_t_span_list = [[] for i in range(len(t_span_list))]
        for i in range(len(t_span_list)):
            if i < len(t_span_list)-1:
                t_span = [t_span_list[i+1][0],t_span_list[i][1]]
                send_t_span_list[i].append(t_span)
        for i in range(len(t_span_list)):
            if i!= 0:
                t_span = [t_span_list[i][0],t_span_list[i-1][1]]
                send_t_span_list[i].append(t_span)
        
        #STEP1-2
        #   in:  t_span_list,cloud_point_list,loss_list 
        #   out: cp_para_list,loss_para_flag_list
        cp_para_list = []
        loss_para_flag_list = []
        for model_id in range(model_num):
            cp_para_item, loss_para_item = \
                get_tspan_cp_loss(cloud_point_list, 
                                 loss_list, 
                                 t_span_list[model_id])
            cp_para_list.append(cp_para_item)
            loss_para_flag_list.append(loss_para_item)

        print_ranked('send_graph',send_graph)
        print_ranked('recv_graph',recv_graph)
        print_ranked('send_t_span_list',send_t_span_list)
        time.sleep(0.1)
        
        #STEP1-3 cloud_point_list,send_t_span_list -> send_x_data_list
        #   Note:in this section XPINNs differ from tpPINNs
        send_xdata_list = [[] for i in range(model_num)]
        for model_id in range(model_num):
            for t_span in send_t_span_list[model_id]:
                cp_para_item, _ = get_tspan_cp_loss(
                    cloud_point_list, 
                    loss_list, 
                    t_span)
                x_data_item_list = [item[0] for item in cp_para_item]
                send_xdata_list[model_id].append(np.vstack(x_data_item_list))   

        print_ranked('send_xdata_list',send_xdata_list)
        time.sleep(0.1)     

        # STEP1-4 cloud_point_list,send_t_span -> send_x_data_list
        recv_xdata_list = [[] for i in range(model_num)]
        recv_ydata_shape_list = [[] for i in range(model_num)] 
        for u,edge_list in enumerate(send_graph):
            for i,v in enumerate(edge_list):
                # 交界区坐标
                recv_xdata_list[v].append(send_xdata_list[u][i].copy())
                # 交界区取值
                recv_ydata_shape_item = list(send_xdata_list[u][i].shape)
                recv_ydata_shape_item[-1] = 1
                recv_ydata_shape_list[v].append(recv_ydata_shape_item)

        print('recv_ydata_shape_list',recv_ydata_shape_list)
        
    # 发送
    local_cp = scatter_objects(cp_para_list if rank == 0 else None)
    local_para_flag = scatter_objects(
        loss_para_flag_list if rank == 0 else None)
    local_send_xdata = scatter_objects(send_xdata_list if rank == 0 else None)
    local_recv_ydata_shape = scatter_objects(
        recv_ydata_shape_list if rank == 0 else None)
    local_recv_xdata = scatter_objects(recv_xdata_list if rank == 0 else None)

    if debug_print_flag:
        print_ranked('cp_para:',local_cp)
        print_ranked('loss_para_flag:',local_para_flag)
        print_ranked('send_xdata',local_send_xdata)
        print_ranked('send_graph',send_graph)
        print_ranked('recv_graph',recv_graph)
        print_ranked('recv_ydata_shape',local_recv_ydata_shape)
        pass
    

    print(f'ok{rank}')
    solver = CloudPointSolver(
        model = [1, 128,128,128, 1], 
        optimizer = "adam",
        device = device)

    def test(x_plot_list,solver):
        """ 测试数据已经在计算节点中保存时调用该函数:[需被封装]
        """
        x_plot = scatter_objects(x_plot_list if rank == 0 else None)
        local_y_test = solver.model_eval(x_plot,use_best_model_flag=False)
        y_test_list = gather_objects(local_y_test)
        return y_test_list
    
    def test_eval(cp_para_list,solver):
        x_test_list = [cp_para[0][0] for cp_para in cp_para_list] \
            if rank == 0 else None
        y_test_list = test(x_test_list, solver)
        if rank == 0:
            for i in range(model_num):
                print_ranked(y_test_list[i].shape)
            x_test_whole = np.concatenate(x_test_list, axis = 0)
            y_test_whole = np.concatenate(y_test_list, axis = 0)
            error = np.mean(np.abs(np.cos(np.pi*x_test_whole) - y_test_whole))
            return y_test_list,error
        else:
            return None,None
        
    _,error = test_eval(cp_para_list if rank == 0 else None,solver)
    if rank ==0: print(error)

    error_list = []

    
    epoch_num = 600
    info_exchange_round = 10
    test_round = 20
    for epoch in range(epoch_num):
        if epoch % info_exchange_round == 0:
            # 采样
            send_ydata = [solver.model_eval(item,use_best_model_flag=False) 
                          for item in local_send_xdata]
            send_ydata = [torch.tensor(item) for item in send_ydata]
            # 通信
            dist.barrier()
            recv_ydata = graph_send_tensor_list(send_graph, recv_graph, 
                send_ydata,
                local_recv_ydata_shape)

        if epoch % test_round == 0:
            _,error = test_eval(cp_para_list if rank == 0 else None,solver)
            if rank ==0: print(epoch, error)

        addition_cp_para = [(i,j.numpy()) for i,j in zip(local_recv_xdata,recv_ydata)]
        addition_loss= [data_loss_factory(loss_fn) for i in addition_cp_para]
        last_cp_para = local_cp + addition_cp_para
        
        loss_filter = \
              [x for x, include in zip(loss_list, local_para_flag) if include]
        last_loss = loss_filter + addition_loss

        solver.train_step(
            cloud_point_list=last_cp_para,
            loss_list = last_loss
        )
    
    plot_y,error = test_eval(cp_para_list if rank == 0 else None,solver)
    if rank==0:
        plot_x = [item[0][0] for item in cp_para_list]
        plot_x = np.concatenate(plot_x, axis = 0)
        plot_y = np.concatenate(plot_y, axis = 0)
        plt.figure()
        plt.scatter(plot_x,plot_y)
        plt.show()

    # for print complete 
    print(f'ok{rank}')
    dist.barrier()
    time.sleep(0.1)

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