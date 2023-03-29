import copy

import numpy as np
import torch

from ..ds import get_data_loader,construct_data_loader_list
from ..get_net import ForwardNetwork
from .. import ds

# TODO: should replace to Solver

class CloudPointSolver():
    """ 该求解器类可根据点云数据与用户定义的损失计算函数进行方程求解

    Attributes:
        _model
        _default_data_loader_list
        _loss_list
        _loss_weight_list
        _optimizer
        _batchsize
        _log
    """

    def __init__(self, cloud_point_list:list = None, loss_list:list = None, 
        *,
        loss_weight_list:list = None,
        model = 'default', 
        optimizer = 'adam', 
        batchsize = 'all', 
        device = 'default'):
        """构造过程将cloud point转换为torch.utils.data.DataLoader并处理其他选项
        
        Args:
        cloud_point_list:
            列表类型.点云数据,每个元素为长度为2的元组,对应一个损失项的输入/输出。
            我们将会根据cloud_point_list确定数据集大小进而确定各数据集batch_size：确
            保len(cloud_point_list[i][0])==len(cloud_point_list[i][1])，len(clou
            d_point_list[i][0])将会是我们确定数据集大小的依据。
        loss_list:
            列表类型.损失函数,类型为可迭代对象,其长度应与cloud_point相同
        loss_weight_list:
            列表类型.损失权重
        model:
            字符串类型,列表类型或已实现的torch.nn.Module网络.
        optimizer:
            字符串类型或torch.optim.Optimizer类型,优化器.
        batchsize:
            'all'或正整数,
            当其为'all'时（默认），其将整个cloud_point作为计算依据.
            训练时最大的cloud_point所对应的batch_size大小，其它cloud_point所
            对应的batch_size大小将自适应调整.
        """
        if device == "default":
            self._device = torch.device("cuda:0" if torch.cuda.is_available()
                else "cpu")
        else:
            self._device = device

        # 无论自定义训练循环还是内置循环都必须赋的初值
        self._log = []
        self._best_model = -1
        self._best_loss = float("inf")

        # deal with model
        if isinstance(model,str):
            raise NotImplementedError
        elif isinstance(model,list):
            self._model = ForwardNetwork(model)
        elif isinstance(model,torch.nn.Module):
            self._model = model
        else:
            raise NotImplementedError
        self._model = self._model.to(self._device)

        # deal with optimizer
        if optimizer == "adam":
            self._optimizer = torch.optim.Adam(self._model.parameters(), \
                lr=1e-3)
        else:
            raise NotImplementedError

        # 未设置值，自定义推进过程
        if cloud_point_list == None:
            return
        else:
            self._default_data_loader_list = construct_data_loader_list(
                cloud_point_list, batchsize, self._device
            )

        # deal with loss function
        self._loss_list = loss_list
        if loss_weight_list == None:
            self._loss_weight_list = \
                torch.ones(len(self._loss_list), device = self._device)
        else:
            self._loss_weight_list = torch.tensor(loss_weight_list)
            self._loss_weight_list.to(self._device)

    def train_step(self, *,
        cloud_point_list = None,
        loss_list = None,
        loss_weight_list = None,
        batchsize = 'all'):
        """使神经网络训练步进一步，要么全指定None，要么全部参数进行设置

        Args:
        cloud_point
        loss_list
        loss_weight_list
            列表类型，约束loss_list被指定时，loss_weight_list必须被指定
            ATTENTION：loss_weight_list存在调用歧义的默认值行为，其可能被理
            解为使用成员对象的值，也可能被理解为根据loss_list长度赋一个全1值。
            所以，我们约束loss_list被指定时，loss_weight_list必须被指定
        batchsize
            列表类型，约束cloud_point_list被指定时，batchsize必须被指定
            ATTENTION：cloud_point_list存在调用歧义的默认值行为，其可能被理
            解为使用成员对象的值，也可能被理解为根据cloud_point_list产生全1值。
            所以，我们约束cloud_point_list被指定时，batchsize必须被指定
        """
        
        all_set_flag = cloud_point_list!=None and \
            loss_list!=None
        all_none_flag = cloud_point_list==None and \
            loss_list==None
        
        if not(all_none_flag) and not(all_set_flag):
            raise(NotImplementedError)
        
        
        # 默认情况处理
        if(all_none_flag):
            used_data_loader_list = self._default_data_loader_list
            used_loss_list = self._loss_list
            used_loss_weight_list = self._loss_weight_list
        if(all_set_flag):
            used_data_loader_list = construct_data_loader_list(
                cloud_point_list,
                batchsize,
                self._device
            )
            used_loss_list = loss_list
            if loss_weight_list == None:
                used_loss_weight_list = \
                    torch.ones(len(used_loss_list), device = self._device)
            

        # 训练开始
        self._model.train()
        num_loss = len(used_loss_list)
        for batch_data_list in zip(*used_data_loader_list):
            # 每一小batch执行
            loss_list = []
            for loss_iter_num in range(num_loss): 
                loss_item = used_loss_list[loss_iter_num]
                loss_weight_item = used_loss_weight_list[loss_iter_num]
                batch_data = batch_data_list[loss_iter_num]
                # 加权求和
                base_loss = loss_item(self._model, batch_data)
                loss_list.append(base_loss * loss_weight_item)
            loss_all = torch.sum(torch.stack(loss_list))
            
            self._optimizer.zero_grad()
            loss_all.backward()
            self._optimizer.step()


    def test_step(self, *,
        cloud_point_list = None,
        loss_list = None,
        loss_weight_list = None,
        batchsize = 'all',
        print_flag = False):
        """进行单步测试

        Args:
        cloud_point:
            用于测试的点云数据
        loss:
            损失函数
        print_flag:
            是否打印测试得到的各个loss情况
        """
        self._model.eval()

        all_set_flag = cloud_point_list!=None and \
            loss_list!=None
        all_none_flag = cloud_point_list==None and \
            loss_list==None
        
        if not(all_none_flag) and not(all_set_flag):
            raise(NotImplementedError)
        
        
        # 默认情况处理
        if(all_none_flag):
            used_data_loader_list = self._default_data_loader_list
            used_loss_list = self._loss_list
            used_loss_weight_list = self._loss_weight_list
        if(all_set_flag):
            used_data_loader_list = construct_data_loader_list(
                cloud_point_list,
                batchsize,
                self._device
            )
            used_loss_list = loss_list
            if loss_weight_list == None:
                used_loss_weight_list = \
                    torch.ones(len(used_loss_list), device = self._device)

        # 计算loss_all
        num_loss = len(used_loss_list)

        # 计算loss_item(shape should be (loss_kind_num * batch_num))
        loss_per_batch = [[] for i in range(num_loss+1)]
        for batch_data_list in zip(*used_data_loader_list):
            # 每一batch依次计算各类loss
            loss_all = torch.tensor([0], device=self._device)    
            for loss_iter_num in range(num_loss): 
                loss_item = used_loss_list[loss_iter_num]
                loss_weight_item = used_loss_weight_list[loss_iter_num]
                batch_data = batch_data_list[loss_iter_num]
                # 加权求和
                base_loss = loss_item(self._model, batch_data)
                loss_all = loss_all + base_loss * loss_weight_item
                # 记录loss情况
                loss_per_batch[loss_iter_num].append(
                    base_loss.detach().cpu().numpy().squeeze())

            # 清除梯度
            self._optimizer.zero_grad()
            loss_all.backward()
            # 保存加权得到的loss
            loss_per_batch[num_loss].append(
                loss_all.detach().cpu().numpy().squeeze())
        
        loss_per_batch = np.array(loss_per_batch)
        loss_entire_epoch = np.mean(loss_per_batch, axis = 1)
        self._log.append(loss_entire_epoch)

        if print_flag == True:
            print('loss is', loss_entire_epoch)

        if self._best_loss > loss_all:
            self._best_loss = loss_all
            self._best_model = copy.deepcopy(self._model)
   

    def model_eval(self, x_point, use_best_model_flag = True):
        """使用最佳模型进行测试

        Args:
        x_point: 计算的坐标位置
        """
        
        x_point = torch.tensor(x_point)
        x_point = x_point.to(torch.float32).to(self._device)
        if use_best_model_flag == True:
            if self._best_model == -1:
                self._model.eval()
                calc_val = self._model(x_point)
            else:
                self._best_model.eval()
                calc_val = self._best_model(x_point)
        else:
            self._model.eval()
            calc_val = self._model(x_point)

        calc_val = calc_val.detach().cpu().numpy()
        return calc_val


    def get_logger(self):
        """返回solver记录的测试日志

        在每次调用
        """
        return self._log