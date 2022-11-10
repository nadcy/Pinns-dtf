from xmlrpc.server import DocXMLRPCRequestHandler
import torch
from torch import nn
class DataNetwork(nn.Module):
    """ 用于有限差分
    """
    def __init__(self, net_size_vec=[2,64,64,64,64,3]):
        super(DataNetwork, self).__init__()
        sqnet_para_list = []
        # like [2,128,64,32,2], iter end in idx 3(32)
        for i in range(len(net_size_vec)-2):
            sqnet_para_list.append(
                nn.Linear(net_size_vec[i],net_size_vec[i+1]),
            )
            sqnet_para_list.append(nn.Tanhshrink())
        sqnet_para_list.append(
            nn.Linear(net_size_vec[-2],net_size_vec[-1])
            )
        self.linear_relu_stack = nn.Sequential(
            *sqnet_para_list
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 输出维度为 5*batch_size*3，其中5代表中心差分五点，3代表3个输出物理量
    def forward(self, x):
        bs = x.shape[0]
        up_x = x + torch.tensor([0,0.05]).to(self.device)
        down_x = x + torch.tensor([0,-0.05]).to(self.device)
        l_x = x + torch.tensor([-0.05,0]).to(self.device)
        r_y = x + torch.tensor([0.05,0]).to(self.device)
        x_vec =  torch.cat((x, up_x, down_x, l_x, r_y), 0)
        y = self.linear_relu_stack(x_vec)
        y = y.split(bs)
        y = torch.stack(y)
        return y

class Data7Network(nn.Module):
    def __init__(self, net_size_vec=[2,64,64,64,64,3]):
        super(Data7Network, self).__init__()
        sqnet_para_list = []
        # like [2,128,64,32,2], iter end in idx 3(32)
        for i in range(len(net_size_vec)-2):
            sqnet_para_list.append(
                nn.Linear(net_size_vec[i],net_size_vec[i+1]),
            )
            sqnet_para_list.append(nn.Tanhshrink())
        sqnet_para_list.append(
            nn.Linear(net_size_vec[-2],net_size_vec[-1])
            )
        self.linear_relu_stack = nn.Sequential(
            *sqnet_para_list
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 输出维度为 7*batch_size*3，其中5代表中心差分五点，3代表3个输出物理量
    def forward(self, x):
        bs = x.shape[0]
        up_x = x + torch.tensor([0,0.05,0]).to(self.device)
        down_x = x + torch.tensor([0,-0.05,0]).to(self.device)
        l_x = x + torch.tensor([-0.05,0,0]).to(self.device)
        r_x = x + torch.tensor([0.05,0,0]).to(self.device)
        before_x = x + torch.tensor([0,0,-0.05]).to(self.device)
        after_x = x + torch.tensor([0,0,0.05]).to(self.device)
        x_vec =  torch.cat((x, up_x, down_x, l_x, r_x, before_x, after_x), 0)
        y = self.linear_relu_stack(x_vec)
        y = y.split(bs)
        y = torch.stack(y)
        return y



class ForwardNetwork(nn.Module):
    def __init__(self, net_size_vec=[2,64,64,64,64,3]):
        super(ForwardNetwork, self).__init__()
        sqnet_para_list = []
        # like [2,128,64,32,2], iter end in idx 3(32)
        for i in range(len(net_size_vec)-2):
            sqnet_para_list.append(
                nn.Linear(net_size_vec[i],net_size_vec[i+1]),
            )
            sqnet_para_list.append(nn.Tanh())
        sqnet_para_list.append(
            nn.Linear(net_size_vec[-2],net_size_vec[-1])
            )
        self.linear_relu_stack = nn.Sequential(
            *sqnet_para_list
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

class IndependentNetwork(nn.Module):
    def __init__(self, net_size_vec=[2,32,32,32,32,1]):
        super(IndependentNetwork, self).__init__()
        sqnet_para_list = []

        for i in range(len(net_size_vec)-2):
            sqnet_para_list.append(
                nn.Linear(net_size_vec[i],net_size_vec[i+1])
            )
            sqnet_para_list.append(nn.Mish())
        sqnet_para_list.append(
            nn.Linear(net_size_vec[-2],net_size_vec[-1])
            )
        self.linear_relu_stack1 = nn.Sequential(
            *sqnet_para_list
        )
        self.linear_relu_stack2 = nn.Sequential(
            *sqnet_para_list
        )
        self.linear_relu_stack3 = nn.Sequential(
            *sqnet_para_list
        )

    def forward(self, x):
        p = self.linear_relu_stack1(x)
        u = self.linear_relu_stack2(x)
        v = self.linear_relu_stack3(x)
        return torch.cat((p, u, v), -1)

def get_model(model_proto, device):
    """ deal with model
    """
    if isinstance(model_proto,str):
        raise NotImplementedError
    elif isinstance(model_proto,list):
        model = ForwardNetwork(model_proto)
    elif isinstance(model_proto,torch.nn.Module):
        model = model_proto
    else:
        raise NotImplementedError
    model = model.to(device)
    return model