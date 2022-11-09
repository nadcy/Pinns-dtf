沿时间方向的区域分解模块的功能分为两部分

代码重构
核心工作：
    对solver类中参数处理的部分进行代码复用。

实现comm_solver类：
额外包含属性/方法
    发送：send,send_spatial_domain_pc
    接收：recv,recv_spatial_domain_pc,comm_loss

对于现有的solver API，可执行以下改进：
需求一
    将solver处理参数的过程进行封装；
需求二
    封装使用loss_func(PDE/data), loss_weight, ds迭代一个epoch的过程为一个无状态函
    数，并在solver中调用这个函数以实现功能。

第一部分：预处理
    核心：根据沿时间方向的区域分解算法处理空间点云数据为时间点云数据。
    损失函数：用户考虑第一维为时间维自己定义。 
    网络训练参数：根据传入参数是否为标量定义。 

第二部分：部署计算
    接口：(solver_train_args_list, ic_cp_only_space, t0_val, overlap_length)
        solver_train_args_list: 每个元素为solver运行需要的参数（初值条件除外）
        ic_cp_only_space：代表空间域的cloud point二维数组
        t0_val: 长度应与ic_cp_only_space相同
        overlap_length
    分解方法。 
