import copy

import numpy as np
import torch

from .ds import get_data_loader
from .get_net import ForwardNetwork

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
        batchsize = 32, 
        device = 'default'):
        """构造过程将cloud point转换为torch.utils.data.DataLoader并处理其他选项
        
        Args:
        cloud_point_list:
            列表类型.点云数据,,每个元素为长度为1或2的元组,当损失对应PDE损
            失是元素通常为xdata,当损失对应data损失时,通常列表元素为(x_data,
            y_data) 
        loss:
            列表类型.损失函数,类型为可迭代对象,其长度应与cloud_point相同
        loss_weight:
            列表类型.损失权重
        model:
            字符串类型,列表类型或已实现的torch.nn.Module网络.
        optimizer:
            字符串类型,字典类型(用作参数列表)或torch.optim.Optimizer类型,优化器.
        batchsize:
            正整数,训练时batch_size大小.
        """
        if device == "default":
            self._device = torch.device("cuda:0" if torch.cuda.is_available()
                else "cpu")
        else:
            raise NotImplementedError

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
            # change cloud point data to dataloader
            self._default_data_loader_list = []
            for i,cloud_point_item in enumerate(cloud_point_list):
                if isinstance(batchsize, int):
                    data_loader_item = get_data_loader(cloud_point_item, \
                        batch_size = batchsize)
                elif isinstance(batchsize, list):
                    data_loader_item = get_data_loader(cloud_point_item, \
                        batch_size = batchsize[i])
                else:
                    raise NotImplementedError
                self._default_data_loader_list.append(data_loader_item)

        # deal with loss_list
        self._loss_list = loss_list
        
        # deal with loss weight list
        if loss_weight_list == None:
            self._loss_weight_list = \
                torch.ones(len(self._loss_list), device = self._device)
        else:
            self._loss_weight_list = torch.tensor(loss_weight_list)
            self._loss_weight_list.to(self._device)
        
        self._batchsize = batchsize


    def train_step(self, *,
        cloud_point_list = None,
        loss_list = None,
        loss_weight_list = None,
        optimizer = None,
        batchsize = None):
        """使神经网络训练步进一步，默认参数值不为None时可自定义训练流程

        Args:
        cloud_point
        loss_list
        loss_weight_list
            列表类型，约束loss_list被指定时，loss_weight_list必须被指定
            ATTENTION：loss_weight_list存在调用歧义的默认值行为，其可能被理
            解为使用成员对象的值，也可能被理解为根据loss_list长度赋一个全1值。
            所以，我们约束loss_list被指定时，loss_weight_list必须被指定
        optimizer
        batchsize
            列表类型，约束cloud_point_list被指定时，batchsize必须被指定
            ATTENTION：cloud_point_list存在调用歧义的默认值行为，其可能被理
            解为使用成员对象的值，也可能被理解为根据cloud_point_list产生全1值。
            所以，我们约束cloud_point_list被指定时，batchsize必须被指定
        """
        # 异常判定
        # TODO 改为xor跟随
        if loss_list==None and loss_weight_list!=None:
            raise(NotImplementedError)
        if cloud_point_list==None and batchsize!=None:
            raise(NotImplementedError)

        self._model.train()

        if cloud_point_list == None:
            used_data_loader_list = self._default_data_loader_list
        else:
            # 使用 cloud_point_list,batch_size 生成 used_data_loader_list
            used_data_loader_list  = []
            for i,cloud_point_item in enumerate(cloud_point_list):
                if isinstance(batchsize, int):
                    data_loader_item = get_data_loader(cloud_point_item, \
                        batch_size = batchsize)
                elif isinstance(batchsize, list):
                    data_loader_item = get_data_loader(cloud_point_item, \
                        batch_size = batchsize[i])
                else:
                    raise NotImplementedError
                used_data_loader_list.append(data_loader_item)

        # 处理 loss_list 与 loss_weight_list 默认值行为（直接赋值即可）
        # ?直接用函数默认值参数实现会不会更好
        # 注意batch_size还没有实现默认值行为
        if loss_list == None:
            used_loss_list = self._loss_list
        else:
            used_loss_list = loss_list

        # 注意：loss_weight_list 从不
        if loss_weight_list != None:
            # 有参数用参数
            used_loss_weight_list = torch.tensor(loss_weight_list)
            used_loss_weight_list = used_loss_weight_list.to(self._device)
        else:
            # 无参数使用默认值
            used_loss_weight_list = self._loss_weight_list

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
        batchsize = None,
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

        # 异常判定
        # TODO 改为xor跟随
        if loss_list==None and loss_weight_list!=None:
            raise(NotImplementedError)
        if cloud_point_list==None and batchsize!=None:
            raise(NotImplementedError)

        # BEGIN: 参数处理
        if cloud_point_list == None:
            used_data_loader_list = self._default_data_loader_list
        else:
            # 使用 cloud_point_list,batch_size 生成 used_data_loader_list
            used_data_loader_list  = []
            for i,cloud_point_item in enumerate(cloud_point_list):
                if isinstance(batchsize, int):
                    data_loader_item = get_data_loader(cloud_point_item, \
                        batch_size = batchsize)
                elif isinstance(batchsize, list):
                    data_loader_item = get_data_loader(cloud_point_item, \
                        batch_size = batchsize[i])
                else:
                    raise NotImplementedError
                used_data_loader_list.append(data_loader_item)
        # 处理 loss_list 与 loss_weight_list 默认值行为（直接赋值即可）
        # ?直接用函数默认值参数实现会不会更好
        # 注意batch_size还没有实现默认值行为
        if loss_list == None:
            used_loss_list = self._loss_list
        else:
            used_loss_list = loss_list
        # 注意：loss_weight_list 从不
        if loss_weight_list != None:
            # 有参数用参数
            used_loss_weight_list = torch.tensor(loss_weight_list)
            used_loss_weight_list = used_loss_weight_list.to(self._device)
        else:
            # 无参数使用默认值
            used_loss_weight_list = self._loss_weight_list

        # 计算loss_all
        num_loss = len(used_loss_list)
        # END: 参数处理

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
   

    def model_eval(self, x_point):
        """使用最佳模型进行测试

        Args:
        x_point: 计算的坐标位置
        """
        x_point = torch.tensor(x_point)
        x_point = x_point.to(torch.float32).to(self._device)
        self._best_model.eval()
        calc_val = self._best_model(x_point)
        calc_val = calc_val.detach().cpu()
        return calc_val


    def get_logger(self):
        """返回solver记录的测试日志

        在每次调用
        """
        return self._log