import os
import time

import copy

import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from user_fun.geom import get_tspan_cp_loss
from user_fun.solver.cp_solver import CloudPointSolver
from user_fun.bc import data_loss_factory
from user_fun.comm import \
    print_ranked,scatter_objects,gather_objects,graph_send_tensor_list

class Trainer:
    def __init__(self, rank, worker_num,
                 t_span_list, algo,
                 print_flag, device,
                 send_graph = None, send_t_span_list = None,
                 eval_func = None):
        
        if worker_num<=2:
            raise(NotImplementedError)
        if algo != 'udPINNs' and algo != 'XPINNs':
            raise(NotImplementedError)

        self.worker_num = worker_num
        self.rank = rank
        self.t_span_list = t_span_list
        self.algo = algo
        self.print_flag = print_flag
        self.device = device
        self.eval_func = eval_func

        self.loss_fn = torch.nn.MSELoss()

        if send_graph == None and send_t_span_list == None:
            self.send_graph = [[] for i in range(worker_num)]
            if algo == 'udPINNs':
                for i in range(worker_num - 1):
                    self.send_graph[i].append(i+1)
            if algo == 'XPINNs':
                for i in range(worker_num - 1):
                    self.send_graph[i].append(i+1)
                for i in range(1,worker_num):
                    self.send_graph[i].append(i-1)

            # STEP1-1 Determine overlapping tpan: t_span_list -> send_t_span_list
            self.send_t_span_list = [[] for i in range(len(t_span_list))]
            if algo == 'udPINNs':
                for i in range(worker_num-1):
                    t_span = [t_span_list[i+1][0],t_span_list[i][1]]
                    self.send_t_span_list[i].append(t_span)
            if algo == 'XPINNs':
                for i in range(worker_num-1):
                    t_span = [t_span_list[i+1][0],t_span_list[i][1]]
                    self.send_t_span_list[i].append(t_span)
                for i in range(1,worker_num):
                    t_span = [t_span_list[i][0],t_span_list[i-1][1]]
                    self.send_t_span_list[i].append(t_span)

            self.recv_graph = [[] for i in range(worker_num)]
            for u,edge_list in enumerate(self.send_graph):
                for i,v in enumerate(edge_list):
                    self.recv_graph[v].append(u)
        else:
            self.send_graph = send_graph
            self.send_t_span_list = send_t_span_list
        
                    
        if print_flag == True:
            if rank == 0:
                print_ranked('send_graph',self.send_graph)
                print_ranked('recv_graph',self.recv_graph)

        
    def assign_task(self,cloud_point_list,loss_list):
        self.loss_list = loss_list
        if self.rank == 0:
            self.cloud_point_list = cloud_point_list
        
            #STEP1-1
            #   in:  t_span_list,cloud_point_list,loss_list 
            #   out: cp_para_list,loss_para_flag_list
            self.cp_para_list = []
            self.loss_para_flag_list = []
            for model_id in range(self.worker_num):
                cp_para_item, loss_para_item = \
                    get_tspan_cp_loss(cloud_point_list, 
                                    loss_list, 
                                    self.t_span_list[model_id])
                self.cp_para_list.append(cp_para_item)
                self.loss_para_flag_list.append(loss_para_item)
            if self.rank == 0 : print(self.cp_para_list)

        
            # STEP1-2 cloud_point_list,send_t_span_list -> send_x_data_list
            # Note:in this section XPINNs differ from tpPINNs
            self.send_xdata_list = [[] for i in range(self.worker_num)]
            for model_id in range(self.worker_num):
                for t_span in self.send_t_span_list[model_id]:
                    cp_para_item, _ = get_tspan_cp_loss(
                        cloud_point_list, 
                        loss_list, 
                        t_span)
                    x_data_item_list = [item[0] for item in cp_para_item]
                    x_data_item_list = np.vstack(x_data_item_list)
                    self.send_xdata_list[model_id].append(x_data_item_list)   

            # STEP1-3 cloud_point_list,send_t_span -> send_x_data_list
            self.recv_xdata_list = [[] for i in range(self.worker_num)]
            self.recv_ydata_shape_list = [[] for i in range(self.worker_num)] 
            for u,edge_list in enumerate(self.send_graph):
                for i,v in enumerate(edge_list):
                    # 交界区坐标
                    self.recv_xdata_list[v].append(
                        self.send_xdata_list[u][i].copy())
                    # 交界区取值
                    recv_ydata_shape_item = \
                        list(self.send_xdata_list[u][i].shape)
                    recv_ydata_shape_item[-1] = 1
                    self.recv_ydata_shape_list[v].append(recv_ydata_shape_item)

            print('recv_ydata_shape_list',self.recv_ydata_shape_list)
        
        self.local_cp = scatter_objects(
            self.cp_para_list if self.rank == 0 else None)
        self.local_para_flag = scatter_objects(
            self.loss_para_flag_list if self.rank == 0 else None)
        self.local_send_xdata = scatter_objects(
            self.send_xdata_list if self.rank == 0 else None)
        self.local_recv_ydata_shape = scatter_objects(
            self.recv_ydata_shape_list if self.rank == 0 else None)
        self.local_recv_xdata = scatter_objects(
            self.recv_xdata_list if self.rank == 0 else None)

        
    def train(self, epoch_num, info_exchange_round, test_round, result_dir,
              model_vec = [1,32,32,32,1]):
        
        print_ranked(self.local_send_xdata)
        dist.barrier()

        loss_record = []
        error_record = []
        test_record = []
        
        self.solver = CloudPointSolver(
            model = model_vec, 
            optimizer = "adam",
            device = self.device)
        
        for epoch in range(epoch_num):
            if epoch % info_exchange_round == 0:
                # sample
                if self.algo == "XPINNs":
                    send_ydata = \
                        [self.solver.model_eval(item,use_best_model_flag=False) 
                            for item in self.local_send_xdata]
                if self.algo == "udPINNs":
                    send_ydata = \
                        [self.solver.model_eval(item,use_best_model_flag=True) 
                            for item in self.local_send_xdata]
                    
                send_ydata = \
                    [torch.tensor(item) for item in send_ydata]
                self.solver._best_model = copy.deepcopy(self.solver._model)
                
                # communicate
                dist.barrier()
                recv_ydata = graph_send_tensor_list(self.send_graph, 
                                                    self.recv_graph, 
                    send_ydata,
                    self.local_recv_ydata_shape)
                
                addition_cp_para = [(i,j.numpy()) for i,j in zip(
                self.local_recv_xdata,recv_ydata)]
                addition_loss= [data_loss_factory(self.loss_fn) 
                                for i in addition_cp_para]
                last_cp_para = self.local_cp + addition_cp_para
                
                loss_filter = \
                    [x for x, include in zip(self.loss_list, self.local_para_flag) 
                    if include]
                last_loss = loss_filter + addition_loss
                
            # TODO 额外保存中间过程
            if epoch % test_round == 0:
                y_test_list = self.test(
                    self.cp_para_list if self.rank == 0 else None, self.solver)
                test_record.append(y_test_list)
                print(epoch)

                # if self.rank ==0: print(epoch, error)

            self.solver.train_step(
                cloud_point_list=last_cp_para,
                loss_list = last_loss
            )
        

        plot_x,plot_y = self.test(
                    self.cp_para_list if self.rank == 0 else None, self.solver)
        if self.rank == 0 and self.eval_func!=None:
            error = self.eval_func(plot_x,plot_y)
            print(error)
        

        if self.rank==0:
            loss_record = self.solver.get_logger()

            # save
            current_directory = os.getcwd()
            data_directory = os.path.join(current_directory, result_dir)
            if not os.path.exists(data_directory):
                os.makedirs(data_directory)

            time_str = datetime.now().strftime('%Y%m%d-%H%M%S')
            COVER_FLAG = False
            if COVER_FLAG == False:
                error_file_name = 'error_record.txt'
                loss_file_name = 'loss_record.txt'
                test_file_name = 'test_record.pkl'

            else:
                error_file_name = time_str + 'error_record.txt'
                loss_file_name = time_str + 'loss_record.txt'
                test_file_name = time_str + 'test_record.pkl'

            
            error_file_path = os.path.join(data_directory, error_file_name)
            loss_file_path = os.path.join(data_directory, loss_file_name)
            test_file_path = os.path.join(data_directory, test_file_name)
            
            
            np.savetxt(error_file_path, error_record)
            np.savetxt(loss_file_path, loss_record)

            import pickle
            with open(test_file_path, 'wb') as file:
                pickle.dump(test_record, file)
        
    def test(self,cp_para_list,solver):
        # 公共
        x_test_list = [cp_para[0][0] for cp_para in cp_para_list] \
            if self.rank == 0 else None
        x_plot = scatter_objects(x_test_list if self.rank == 0 else None)
        local_y_test = solver.model_eval(x_plot,use_best_model_flag=False)
        y_test_list = gather_objects(local_y_test)
        if self.rank == 0:
            x_test_whole = np.concatenate(x_test_list, axis = 0)
            y_test_whole = np.concatenate(y_test_list, axis = 0)
        else:
            x_test_whole = None
            y_test_whole = None
        return x_test_whole,y_test_whole
    
        