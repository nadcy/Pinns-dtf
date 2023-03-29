"""run.py:"""
#!/usr/bin/env python
import time
import os
import sys
sys.path.append('./')
print(sys.path)

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
np.set_printoptions(threshold=8)
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from user_fun.solver.cp_solver import CloudPointSolver
from user_fun.baseline import sint
from user_fun.bc import data_loss_factory

print(torch.__version__)
print(torch.version.cuda)
debug_print_flag = True
save_dir = './timepde/multi_gpu/'

def run(rank, size):
    print(f'rank:{rank},size:{size}:ready')
    loss_fn = torch.nn.MSELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    """
    使用逻辑数组表示每个子区域以及通信区
    例如 (x_data[:,-1]>=t_span[0]) & (x_data[:,-1]<=t_span[1])
    """

    # 原始问题定义
    cloud_point_list,loss_list = sint.get_long_time_problem()
    model_num = size
    # 拓扑图定义
    send_graph = [[1],[2],[]] #全局副本


    recv_graph = [[] for i in range(model_num)] #全局副本
    for u,edge_list in enumerate(send_graph):
        for i,v in enumerate(edge_list):
            recv_graph[v].append(u)

    if rank == 0:
        #STEP1 确定tppinn在时间方向上的分区与重叠区
        # 问题相关代码
        t_span_list = np.array([[0,2.2],[2,4.2],[4,6]])
        send_t_span = [None for i in range(len(t_span_list))]
        for i in range(len(t_span_list)):
            if i == len(t_span_list)-1:
                send_t_span[i] = []
            else:
                send_t_span[i] = [t_span_list[i+1][0],t_span_list[i][1]]
        if debug_print_flag:
            print('send_t_span', send_t_span)

        #STEP2 确定cp数组以及loss-mask数组
        def get_span_cp_loss(cloud_point_list,loss_list,t_span):
            """ 专用于处理时间相关PDE的求解
                para cloud_point_list: 

            """
            res_cloud_point_list = []
            res_loss_list = \
                torch.tensor([False]*len(loss_list), dtype = torch.bool)

            if not(t_span is None):
                for i in range(len(loss_list)):
                    x_data,y_data = cloud_point_list[i]
                    idx = (x_data[:,-1]>=t_span[0]) & (x_data[:,-1]<=t_span[1])
                    choose_x_data = x_data[idx,:]
                    choose_y_data = y_data[idx,:]

                    if len(choose_x_data)!=0:
                        res_cloud_point_list.append(
                            [choose_x_data, choose_y_data]
                            )
                        res_loss_list[i] = True

            return res_cloud_point_list,res_loss_list
            
        cp_para_list = []
        loss_para_flag_list = []
        for model_id in range(model_num):
            cp_para_item, loss_para_item = \
                get_span_cp_loss(cloud_point_list, 
                                 loss_list, 
                                 t_span_list[model_id])
            cp_para_list.append(cp_para_item)
            loss_para_flag_list.append(loss_para_item)
        
        # 交界区计算
        send_xdata_list = [[] for i in range(model_num)]
        for model_id in range(model_num):
            if model_id != model_num - 1:
                cp_para_item, _ = get_span_cp_loss(
                    cloud_point_list, 
                    loss_list, 
                    send_t_span[model_id])
                x_data_item_list = [item[0] for item in cp_para_item]
                send_xdata_list[model_id] = [np.vstack(x_data_item_list)]

        # 图收发
        recv_xdata_list = [[] for i in range(model_num)]
        recv_ydata_shape_list = [[] for i in range(model_num)] 
        for u,edge_list in enumerate(send_graph):
            for i,v in enumerate(edge_list):
                # 交界区坐标
                recv_xdata_list[v].append(send_xdata_list[u][i].copy())
                # 交界区取值
                recv_ydata_shape_item = list(send_xdata_list[u][i].shape)
                recv_ydata_shape_item[1] = 1
                recv_ydata_shape_list[v].append(recv_ydata_shape_item)

        print('recv_ydata_shape_list',recv_ydata_shape_list)
        
    # 通信
    def scatter_objects(scatter_object_list):
        local_obj = [None]
        dist.scatter_object_list(local_obj, scatter_object_list, src=0)
        return local_obj[0]
    
    def gather_objects(local_obj):
        current_rank = dist.get_rank()
        world_size = dist.get_world_size()
        gathered_objects = [None] * world_size if current_rank == 0 else None
        dist.gather_object(local_obj, gathered_objects, dst=0)
        return gathered_objects if current_rank == 0 else None
    
    cp_para = scatter_objects(cp_para_list if rank == 0 else None)
    loss_para_flag = scatter_objects(
        loss_para_flag_list if rank == 0 else None)
    send_xdata = scatter_objects(send_xdata_list if rank == 0 else None)
    recv_ydata_shape = scatter_objects(
        recv_ydata_shape_list if rank == 0 else None)
    recv_xdata = scatter_objects(recv_xdata_list if rank == 0 else None)

    def print_ranked(*args, **kwargs):
        message = "".join(str(arg) for arg in args)
        print(f"[Rank {dist.get_rank()} {message}]\n")

    if debug_print_flag:
        print_ranked('cp_para:',cp_para)
        print_ranked('loss_para_flag:',loss_para_flag)
        print_ranked('send_xdata',send_xdata)
        print_ranked('send_graph',send_graph)
        print_ranked('recv_graph',recv_graph)
        print_ranked('recv_ydata_shape',recv_ydata_shape)
        pass
    

    print(f'ok{rank}')
    solver = CloudPointSolver(model = [1, 64,64,64, 1], optimizer = "adam")

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
    """ 在每个worker内将
    """

    def graph_send_tensor_list(
            send_graph, recv_graph, 
            send_tensor_list,
            recv_tensor_shape):
        """ 在全局中使用，根据收发拓扑图已经
            para: send_id 反映发送方的拓扑
            para: recv_id 反映接收方的拓扑
            para: send_tensor_list 发送方消息队列
            return: 接收方消息队列（顺序由接收方拓扑决定）
        """
        rank = dist.get_rank()
        req_list = []
        recv_tensor_list = \
            [torch.zeros(item) for item in recv_tensor_shape]
        for send_id_item,send_tensor_item in \
            zip(send_graph[rank],send_tensor_list):
            req_item = dist.isend(send_tensor_item, dst = send_id_item)
            req_list.append(req_item)

        # 发送张量
        for i,recv_id_item in enumerate(recv_graph[rank]):
            req_item = dist.irecv(recv_tensor_list[i], src = recv_id_item)
            req_list.append(req_item)

        # 接收张量
        for req_item in req_list:
            req_item.wait()
        return recv_tensor_list
    

    epoch_num = 600
    info_exchange_round = 10
    test_round = 20
    for epoch in range(epoch_num):
        if epoch % info_exchange_round == 0:
            # 采样
            send_ydata = [solver.model_eval(item) for item in send_xdata]
            send_ydata = [torch.tensor(item) for item in send_ydata]
            # 通信
            recv_ydata = graph_send_tensor_list(send_graph, recv_graph, 
                send_ydata,
                recv_ydata_shape)

        if epoch % test_round == 0:
            _,error = test_eval(cp_para_list if rank == 0 else None,solver)
            if rank ==0: print(epoch, error)
            dist.barrier()

        # 常规训练
        addition_cp_para = [(i,j.numpy()) for i,j in zip(recv_xdata,recv_ydata)]
        addition_loss= [data_loss_factory(loss_fn) for i in addition_cp_para]
        last_cp_para = cp_para + addition_cp_para
        
        loss_filter = \
              [x for x, include in zip(loss_list, loss_para_flag) if include]
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


    # # 首轮误差估值
    time.sleep(0.1)
    print(f'ok{rank}')
    dist.barrier()
    pass
        


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