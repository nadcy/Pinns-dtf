一，优化batch_size默认参数
    目前需要用户保证对于各个loss，dataset_size/batch_size为一个常数，但实际上由于PINNs
    网络规模特征，常用整个数据集进行训练，于是改变该默认行为。
    对于解域过大需设置batch_size的情况，各个cloud_point_list的规模可能不一致，我们以"最大
    的cloud_point_list长度/batch_size"作为每轮迭代的数量自动确定其它输入的标量信息。

二，整合问题情况
    算法测试样例当作benchmark加进框架内。
    热传导，NS等常见方程损失也当作benchmark加进框架内。

三，添加符号定义（废弃）
    网页版再考虑这个事情。

三，添加COMSOL仿真数据导入方法
    可直接添加COMSOL进行训练前测试。