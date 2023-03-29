""" loss的数据集制作
    注意检查：torch.utils.data.TensorDataset
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PointCloudPairDataset(Dataset):
    """ 为(xdata,ydata)成对地构造数据集

    该函数被：get_data_loader依赖
    若一个epoch内使用点数为n，模型输入输出大小为m_in,m_out
    xdata(shape = n * m_in, type = ndarray)
    ydata(shape = n * m_out, type = ndarray)

    Attributes:
        xdata: xdata(shape = n * m_in, type = ndarray)
        ydata: ydata(shape = n * m_out, type = ndarray)
    """
    def __init__(self, xdata,ydata,device_str):
        self.xdata = torch.tensor(xdata,dtype = torch.float32, \
            device = device_str)
        self.ydata = torch.tensor(ydata,dtype = torch.float32, \
            device = device_str)

    def __len__(self):
        return len(self.xdata)

    def __getitem__(self, idx):
        resx = self.xdata[idx]
        resy = self.ydata[idx]
        return resx,resy

class PointCloudSingleDataset(Dataset):
    """ 为xdata构造数据集

    该函数被：get_data_loader依赖
    若一个epoch内使用点数为n，模型输入输出大小为m_in,m_out
    xdata(shape = n * m_in, type = ndarray)

    Attributes:
        xdata: xdata(shape = n * m_in, type = ndarray)
    """
    def __init__(self, xdata,device_str):
        self.xdata = torch.tensor(xdata,dtype = torch.float32, \
            device = device_str)

    def __len__(self):
        return len(self.xdata)

    def __getitem__(self, idx):
        resx = self.xdata[idx]
        return resx

def get_data_loader(data_list, batch_size, device_str):
    """ 构造一个data_loader
    """
    if len(data_list) == 1:
        x_data = data_list[0]
        data_set = PointCloudSingleDataset(x_data, device_str)
    elif len(data_list) == 2:
        x_data, y_data = data_list
        data_set = PointCloudPairDataset(x_data, y_data, device_str)

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader


def construct_data_loader_list(cloud_point_list, batchsize, device_str):
    """ 构造data_loader_list
    """
    dataset_size_list = [len(item[0]) for item in cloud_point_list]
    
    # 计算batchsize_list
    if batchsize == 'all':
        batchsize_list = dataset_size_list
    else:
        if max(dataset_size_list)%batchsize == 0:
            batch_num = max(dataset_size_list)//batchsize
        else:
            batch_num = max(dataset_size_list)//batchsize + 1
        batchsize_list = []
        for i,cloud_point_item in enumerate(cloud_point_list):
            if dataset_size_list[i]%batch_num == 0:
                batchsize_list.append(dataset_size_list[i]//batch_num)
            else:
                batchsize_list.append(dataset_size_list[i]//batch_num+1)

    # 计算data_loader_list
    data_loader_list = []
    for i,cloud_point_item in enumerate(cloud_point_list):
        data_loader_item = get_data_loader(cloud_point_item, 
            batch_size = batchsize_list[i],
            device_str = device_str)
        data_loader_list.append(data_loader_item)
    return data_loader_list
