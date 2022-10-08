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

def get_data_loader(data_list, batch_size = 64, device_str = None):
    if device_str == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() \
            else "cpu")
    
    if len(data_list) == 1:
        x_data = data_list[0]
        data_set = PointCloudSingleDataset(x_data, device)
    elif len(data_list) == 2:
        x_data, y_data = data_list
        data_set = PointCloudPairDataset(x_data, y_data, device)

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader