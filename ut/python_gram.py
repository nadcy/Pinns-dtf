import os
print('运行目录',os.getcwd())
import sys
print('当前脚本目录',sys.path[0])

print('解释器目录',sys.executable)
add_path = os.path.join(sys.path[0],'../')
print('添加目录', add_path)

import torch
print(2 * torch.acos(torch.tensor(0.0)))

sys.path.append(add_path)

with open('a.txt','w') as f:
    pass

from user_fun.ds import get_data_loader
from user_fun.ds import PointCloudSingleDataset
import user_fun.ds 
print(user_fun.ds.__C)