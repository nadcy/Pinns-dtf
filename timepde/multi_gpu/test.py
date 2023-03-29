import numpy as np

# 区间的范围和数量
start = 0
end = 6
num_intervals = 5

# 计算每个小区间的宽度
interval_width = (end - start) / (num_intervals) *0.2


# 初始化小区间的起始点
interval_starts = np.linspace(start, end , num_intervals+1)

# 计算每个小区间的结束点
interval_ends = interval_starts  + (end - start) / (num_intervals)+ interval_width

# 结合起始点和结束点，创建形式为 [[start, end], ...] 的结果
t_span = np.column_stack((interval_starts, interval_ends))

# 输出结果
print(t_span)