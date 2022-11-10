""" 

设计思路：
    在PINN训练的过程中，loss的实时变化会导致训练极难收敛，故PINN通常采取的是
    完整epoch送入。我们虽不假设我们solver只处理完整epoch送入的情况，但我们也
    不会对实时变化的loss进行任何优化。
    另外，在10月版本的实践中，我们并不认为solver需要有默认迭代行为的储存机制。
"""
import copy

import numpy as np
import torch

from ..ds import get_data_loader
from ..get_net import ForwardNetwork
from .. import ds

def get_device(device_proto):
    if device_proto == "default":
        return torch.device("cuda:0" if torch.cuda.is_available()
            else "cpu")
    else:
        raise NotImplementedError

def get_dataset(cloud_points,batchsize,device):
    data_loader_item = get_data_loader(cloud_points, batch_size = batchsize)
    return get_data_loader(cloud_points, batch_size = batchsize)

class Loss():
    """ 定义一个神经网络损失项

    Attributes:
        device
        ds
        fun
        weight
    """
    def __init__(self, cloud_points, fun,
        batchsize = None,
        weight = 1,
        device = "default"):

        self.device = get_device(device)

        if batchsize == None:
            if isinstance(cloud_points,list):
                batchsize = cloud_points[0].shape[0]
            else:
                raise NotImplementedError
        self.ds = get_data_loader(
            cloud_points,
            batch_size = batchsize,
            device_str = device
        )

        self.fun = fun
        if weight == 1:
            self.weight = torch.ones(1, device = device)
        else:
            self.weight = weight


class Solver():
    """ 该求解器类可根据点云数据与用户定义的损失计算函数进行方程求解

    Attributes:
        # 训练相关
        model
        loss_list
        optimizer
        device
        # 记录相关
        log
        best_model
    """
    def __init__(self,
        loss_list:list = None,
        model = 'default', 
        optimizer = 'adam', 
        device = 'default'):
        """构造过程将cloud point转换为torch.utils.data.DataLoader并处理其他选项
        
        Args:
        loss_list:
            列表类型.
        model:
            字符串类型,列表类型或已实现的torch.nn.Module网络.
        optimizer:
            字符串类型,字典类型(用作参数列表)或torch.optim.Optimizer类型,优化器.
        """
        self.device = get_device(device)

        # 无论自定义训练循环还是内置循环都必须赋的初值
        self.log = []
        self.best_model = -1
        self.best_loss_val = float("inf")
        self.model = self._model.to(self._device)

        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), \
                lr=1e-3)
        else:
            raise NotImplementedError

        self.loss_list = loss_list

    def train_step(self):
        """使神经网络训练步进一步，默认参数值不为None时可自定义训练流程

        Args:
        loss_list
        optimizer
        """
        self.model.train()

        for batch_idx in len(self.loss_list[0].dataloader):
            loss_list = []
            for loss_idx in len(self.loss_list):
                batch_data  = self.loss_list[loss_idx].ds[batch_idx]
                batch_loss_fun = self.loss_list[loss_idx].fun
                batch_loss_weight = self.loss_list[loss_idx].weight
                base_loss = batch_loss_fun(self.model, batch_data)
                loss_list.append(base_loss * batch_loss_weight)
            loss_all = torch.sum(torch.stack(loss_list))

            self._optimizer.zero_grad()
            loss_all.backward()
            self._optimizer.step()
    
    def test_step(self, print_flag = False):
        num_loss = len(self.loss_list)
        loss_per_batch = [[] for i in range(num_loss+1)]

        self.model.eval()
        for batch_idx in len(self.loss_list[0].dataloader):
            loss_list = []
            for loss_idx in len(self.loss_list):
                batch_data  = self.loss_list[loss_idx].ds[batch_idx]
                batch_loss_fun = self.loss_list[loss_idx].fun
                batch_loss_weight = self.loss_list[loss_idx].weight
                # 加权求和
                base_loss = batch_loss_fun(self.model, batch_data)
                loss_list.append(base_loss * batch_loss_weight)
                # 记录loss情况
                loss_per_batch[loss_idx].append(
                    base_loss.detach().cpu().numpy().squeeze())

            loss_all = torch.sum(torch.stack(loss_list))

            self._optimizer.zero_grad()
            loss_all.backward() # 尝试消除重复建图可能导致的内存泄露

            loss_per_batch[num_loss].append(
                loss_all.detach().cpu().numpy().squeeze())

        # 保存输出相关测试结果
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
