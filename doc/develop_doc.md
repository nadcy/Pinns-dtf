# 超参调优
flow/performance.md 有相关的测试结果记录
# 前后端分离
## 目标
Sample Input:
```
diff(y,x) == 0
```

Sample Output:  
```
def pde_loss(model, data):  
    x_in,y_real = data  
    x_in.requires_grad=True  
    U = model(x_in)  
    u = U[:,[0]]  
    dudx = grad(u, x_in)[0]  
    du2dx2 = grad(dudx, x_in)[0]  
    loss = -du2dx2 - (torch.pi ** 2) * torch.sin(torch.pi * x_in)  
    loss = loss_fn(loss, y_real)  
    return loss  
```  

功能要求：  
1. 重复子图消除  

## 实现方法
途径一：在python代码层上优化  
途径二：在AST抽象语法层上优化  
途径三：在神经网络静态图上优化  
结论11.4：理论上三个途径都应该实现，同时还需要提供额外的符号数学接口

在不考虑复杂功能的情况下，使编码尽可能简单应该使用途径一  
！在当前阶段应不考虑词法分析相关，仅对求导过程进行抽象。