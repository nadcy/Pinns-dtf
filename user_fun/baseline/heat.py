import numpy as np
# reference:
#   https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/heat.html
# solve the PDE:
#   du/dt = a*(du2/dx2) in X[0,L] * T[0,t_max]
#   init condition: u(x,0) = sin(n*pi*x)/L
#   boundary condition = u(0,t) = 0
#   boundary condition = u(L,t) = 0
#   pi = 3.1415...(pi is a CONSTANT)
#   
# solution
#   np.exp(-(n**2 * np.pi**2 * a * t) / (L**2)) * np.sin(n * np.pi * x / L)
#   
# problem parameter
#   a: thermal diffusivity constant.(tunning this parameter increase heat
#       transfer time)
#   L: normalize the sin function to make the x interval is the multiple of half
#       sin cycle.(make sure use default val 1).
#   n: the num of half sin cycle.
#
# recommend problem parameter(use in parallel model test)
#   a = 0.02; L = 1; t_max = 3

class HeatBenchMark():
    def __init__(self, n = 4, a = 0.005, L = 1, tmin = 0, tmax = 1):
        self.n = n
        self.a = a
        self.L = L
        self.tmin = tmin
        self.tmax = tmax

    def heat_eq_exact_solution(self, x, t):
        """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

        Parameters
        ----------
        x : np.ndarray
        t : np.ndarray
        """
        return np.exp(-(self.n**2 * np.pi**2 * self.a * t) / (self.L**2)) * np.sin(self.n * np.pi * x / self.L)

    def gen_exact_solution(self):
        """Generates exact solution for the heat equation for the given values
           of x and t.
        """
        # Number of points in each dimension:
        x_dim, t_dim = (128, 128)

        t = np.linspace(self.tmin, self.tmax, num=t_dim)
        x = np.linspace(0, self.L, num=x_dim)
        tt, xx = np.meshgrid(t, x, indexing='ij')

        usol = np.zeros((t_dim, x_dim)).reshape(t_dim, x_dim)

        # Obtain the value of the exact solution for each generated point:
        for i in range(t_dim):
            for j in range(x_dim):
                usol[i][j] = self.heat_eq_exact_solution(x[j], t[i])

        return tt,xx,usol

    def gen_testdata(self):
        """Generate test data"""
        tt, xx, exact = self.gen_exact_solution()
        X = np.vstack((np.ravel(tt), np.ravel(xx))).T # use T because meshgrid
        y = exact.flatten()[:, None]
        return X, y
    
    def gen_bc_data(self, x_data):
        pass


import torch
from torch import nn
from user_fun.geom import line_linspace,generate_points_in_rectangle
from user_fun.pde import diff
from user_fun import bc
# 热传导继续实现
def HeatBenchMark_longtime():

    density = 32
    init_input = line_linspace([0,0],[1,0],density*2)
    init_output = np.sin(2*np.pi *init_input[:,[0]])

    left_input = line_linspace([0,0],[0,3],density*3)
    left_output = np.sin(np.pi *left_input[:,[1]])
    right_input = line_linspace([1,0],[1,3],density*3)
    right_output = np.sin(np.pi *right_input[:,[1]])

    field_input = generate_points_in_rectangle([0,0],[1,3],density*density*3)
    field_output = np.zeros((field_input.shape[0],1))

    cp_list = [
        [field_input, field_output],
        [left_input, left_output],
        [right_input, right_output],
        [init_input, init_output]
    ]

    a = 0.02
    loss_fn = nn.MSELoss()
    def heat_loss(model, data):
        input,output = data
        input.requires_grad=True
        

        # 数据提取
        x = input[:,[0]]
        t = input[:,[1]]
        use_input = torch.cat([x,t],dim = 1)
        U = model(use_input)
        u = U[:,[0]]

        # 计算一阶导
        dudx = diff(u, x)
        dudt = diff(u, t)
        # 计算二阶导
        du2dx2 = diff(dudx, x)

        loss = dudt - a * du2dx2
        loss = loss_fn(loss, output)
        return loss
    
    left_loss = bc.data_loss_factory(loss_fn)
    right_loss = bc.data_loss_factory(loss_fn)
    init_loss = bc.data_loss_factory(loss_fn)
    loss_list = [
        heat_loss, left_loss, right_loss, init_loss
    ]

    return cp_list,loss_list
