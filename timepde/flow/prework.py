import os
import time

import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata

from user_fun.io import read_comsol
from user_fun.solver.cp_solver import CloudPointSolver
from user_fun.bc import data_loss_factory
from user_fun.comm import \
    print_ranked,scatter_objects,gather_objects,graph_send_tensor_list

def get_xspan_cp_loss(cloud_point_list,loss_list,x_span):
    """
    This function processes time-dependent PDE solutions by filtering
    cloud points and loss values based on a specified time span.

    Args:
        cloud_point_list (list): A list of cloud points, where the last column
                of the input component represents time.
        loss_list: A list of loss values corresponding to each cloud_point.
        t_span (tuple): A tuple with two elements (start_time, end_time) 
                defining the time span.
    Returns:
        res_cloud_point_list: A filtered list of cloud points within the
                                        specified time span.
        res_loss_list: A boolean tensor indicating which loss  values correspond
            to the filtered cloud points.

    Note:
        In cloud_point_list, the last column of the input component represents 
        time.
    """

    res_cloud_point_list = []
    res_loss_flag_list = torch.tensor([False]*len(loss_list), dtype = torch.bool)

    if not(x_span is None):
        for i in range(len(loss_list)):
            x_data,y_data = cloud_point_list[i]
            idx = (x_data[:,0]>=x_span[0]) & (x_data[:,0]<=x_span[1])
            choose_x_data = x_data[idx,:]
            choose_y_data = y_data[idx,:]

            if len(choose_x_data)!=0:
                res_cloud_point_list.append(
                    [choose_x_data, choose_y_data]
                    )
                res_loss_flag_list[i] = True

    return res_cloud_point_list,res_loss_flag_list

class Trainer:
    def __init__(self,rank,worker_num, t_span_list,algo,print_flag,device):
        if worker_num<=2:
            raise(NotImplementedError)
        if algo != 'udPINNs' and algo != 'XPINNs':
            raise(NotImplementedError)

        self.worker_num = worker_num
        self.rank = dist.get_rank()
        self.t_span_list = t_span_list
        self.algo = algo
        self.print_flag = print_flag
        self.device = device

        self.loss_fn = torch.nn.MSELoss()

        self.send_graph = [[] for i in range(worker_num)]
        if algo == 'udPINNs':
            for i in range(worker_num - 1):
                self.send_graph[i].append(i+1)
        if algo == 'XPINNs':
            for i in range(worker_num - 1):
                self.send_graph[i].append(i+1)
            for i in range(1,worker_num):
                self.send_graph[i].append(i-1)

        #STEP1-1 Determine overlapping tpan: t_span_list -> send_t_span_list
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
                    get_xspan_cp_loss(cloud_point_list, 
                                    loss_list, 
                                    self.t_span_list[model_id])
                self.cp_para_list.append(cp_para_item)
                self.loss_para_flag_list.append(loss_para_item)

        
            #STEP1-2 cloud_point_list,send_t_span_list -> send_x_data_list
            #   Note:in this section XPINNs differ from tpPINNs
            self.send_xdata_list = [[] for i in range(self.worker_num)]
            for model_id in range(self.worker_num):
                for t_span in self.send_t_span_list[model_id]:
                    cp_para_item, _ = get_xspan_cp_loss(
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
                    recv_ydata_shape_item[-1] = 3
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

    def read_real(self, path):
        tb = read_comsol.comsol_read(path)
        self.plot_input = tb.values[:,0:2]
        self.plot_p = tb.values[:,2]
        self.plot_u = tb.values[:,3]
        self.plot_v = tb.values[:,4]

        
    def train(self, epoch_num, info_exchange_round, test_round, result_dir):
        
        dist.barrier()

        loss_record = []
        error_record = []
        
        self.solver = CloudPointSolver(
            model = [2, 128,128,128, 3], 
            optimizer = "adam",
            device = self.device)
        
        for epoch in range(epoch_num):
            if epoch % info_exchange_round == 0:
                # sample
                send_ydata = \
                    [self.solver.model_eval(item,use_best_model_flag=False) 
                        for item in self.local_send_xdata]
                send_ydata = [torch.tensor(item) for item in send_ydata]
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
        
                
            if epoch % test_round == 0:
                _,error = self.test_eval(
                    self.cp_para_list if self.rank == 0 else None, self.solver)
                error_record.append(error)
                if self.rank ==0: print(epoch, error)

            self.solver.train_step(
                cloud_point_list=last_cp_para,
                loss_list = last_loss
            )
        
        plot_y,error = self.test_eval(self.cp_para_list 
                                      if self.rank == 0 else None,self.solver)
        

        if self.rank==0:
            loss_record = self.solver.get_logger()

            plot_input = [item[0][0] for item in self.cp_para_list]
            plot_input = np.concatenate(plot_input, axis = 0)
            plot_output = np.concatenate(plot_y, axis = 0)
            plot_u = plot_output[:,1]

            plot_x,plot_y = plot_input[:,0],plot_input[:,1]
            fig, ax = plt.subplots()
            triang = tri.Triangulation(plot_x,plot_y)

            # Mask off unwanted triangles.
            tri_centroid_x = plot_x[triang.triangles].mean(axis=1)
            tri_centroid_y = plot_y[triang.triangles].mean(axis=1)
            mask = np.hypot(tri_centroid_x-1.3,tri_centroid_y-0) < 0.3
            triang.set_mask(mask )
            cm = ax.tripcolor(triang, plot_u, cmap='viridis')

            cbar = ax.figure.colorbar(cm, ax=ax)
            ax.axis('equal')
            plt.show()


            current_directory = os.getcwd()
            data_directory = os.path.join(current_directory, result_dir)
            if not os.path.exists(data_directory):
                os.makedirs(data_directory)

            file_path = os.path.join(data_directory, 'records.npz')
            np.savez(file_path, error_record=error_record, 
                     loss_record=loss_record)
        
    def test(self,x_plot_list,solver):
        x_plot = scatter_objects(x_plot_list if self.rank == 0 else None)
        local_y_test = solver.model_eval(x_plot,use_best_model_flag=False)
        y_test_list = gather_objects(local_y_test)
        return y_test_list
    
    def test_eval(self,cp_para_list,solver):
        x_test_list = [cp_para[0][0] for cp_para in cp_para_list] \
            if self.rank == 0 else None
        y_test_list = self.test(x_test_list, solver)
        if self.rank == 0:

            
            x_test_whole = np.concatenate(x_test_list, axis = 0)
            y_test_whole = np.concatenate(y_test_list, axis = 0)

            p_test_whole = y_test_whole[:,0]
            u_test_whole = y_test_whole[:,1]
            v_test_whole = y_test_whole[:,2]

            plot_input,idx = \
                np.unique(self.plot_input, axis=0,return_index=True)
            
            p_real_whole = griddata(
                plot_input, self.plot_p[idx], x_test_whole, method='cubic',
                fill_value = True)
            u_real_whole = griddata(
                plot_input, self.plot_u[idx], x_test_whole, method='cubic',
                fill_value = True)
            v_real_whole = griddata(
                plot_input, self.plot_v[idx], x_test_whole, method='cubic',
                fill_value = True)
        
        
            p_error = np.mean(np.abs((p_real_whole-p_test_whole))) \
                /np.std(p_real_whole)
            u_error = np.mean(np.abs((u_real_whole-u_test_whole))) \
                /np.std(u_real_whole)
            v_error = np.mean(np.abs((v_real_whole-v_test_whole))) \
                /np.std(v_real_whole)
            
            error = np.array([p_error,u_error,v_error])
            return y_test_list,error
        else:
            return None,None
    
        