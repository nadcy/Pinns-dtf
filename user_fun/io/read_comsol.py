import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def comsol_read(filepath, skiprows = 9):
    def replace_comma(s):
        stack_cnt = 0
        i = 0
        result = ''
        while i < len(s):
            if s[i] == '(':
                stack_cnt = stack_cnt + 1
            elif s[i] == ')':
                stack_cnt = stack_cnt - 1    
            if stack_cnt >0 and s[i] == ',':
                result = result + ';'
            else:
                result = result + s[i]
            i = i + 1
        return result
    
    with open(filepath, 'r') as f:
        for i in range(skiprows):
            line = f.readline()
        columns = replace_comma(line.strip())

    return pd.read_csv(filepath
                       , skiprows=skiprows, delimiter=',', names = columns.split(','))

def process_comsol_time_table(data, space_dimension = 2):
    """ 处理包含时间项的COMSOL仿真结果
    """
    seg_list_dict = dict()
    time_pattern = r'@ t=([\d\.]+)'
    name_pattern = r'^(.*) @'
    first_para_name = re.search(name_pattern, data.columns[2]).group(1)
    for i in range(space_dimension,len(data.columns)):
        point_table = data.iloc[:,i].values
        name = data.columns[i]

        time = float(re.search(time_pattern, name).group(1))
        para_name = re.search(name_pattern, name).group(1)
        
        # 特殊轮次：首个变量所代表的列
        if para_name in data.columns[space_dimension]:
            if i == space_dimension:
                seg_list_dict['x'] = [data.iloc[:,0].values]
                if space_dimension == 2:
                    seg_list_dict['y'] = [data.iloc[:,1].values]
                seg_list_dict['t'] = [np.zeros((len(point_table)))]
                seg_list_dict['t'][0] = seg_list_dict['t'][0] + time
            else:
                seg_list_dict['x'].append(data.iloc[:,0].values)
                if space_dimension == 2:
                    seg_list_dict['y'].append(data.iloc[:,1].values)
                item = np.zeros(len(point_table)) + time
                seg_list_dict['t'].append(item)
        
        if para_name not in seg_list_dict:
            seg_list_dict[para_name] = []
        seg_list_dict[para_name].append(point_table)

    for para_name in seg_list_dict.keys():
        seg_list_dict[para_name] = np.concatenate(seg_list_dict[para_name])
    point_table = pd.DataFrame(seg_list_dict)
    return point_table