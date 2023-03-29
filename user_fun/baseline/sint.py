import torch
import numpy as np
from ..field import D1Field
from ..pde import grad
pde_epoch_size = 16

pi = 2 * torch.acos(torch.tensor(0.0))
def get_problem(pde_epoch_size = 16, loss_fn = torch.nn.MSELoss()):
    pde_input = D1Field([-1,1]).get_field_rand(pde_epoch_size)
    pde_output = np.zeros([pde_epoch_size,1])

    pi = 2 * torch.acos(torch.tensor(0.0))
    def pde_loss(model, data):
        x_in,y_real = data
        x_in.requires_grad=True
        U = model(x_in)
        u = U[:,[0]]
        dudx = grad(u, x_in)[0]
        du2dx2 = grad(dudx, x_in)[0]
        loss = -du2dx2 - (pi ** 2) * torch.sin(pi * x_in)
        loss = loss_fn(loss, y_real)
        return loss
    
    ## define a bc
    bc_epoch_size = 2
    bc_input = np.array([[-1],[1]])
    bc_output = np.zeros([bc_epoch_size ,1])

    from user_funn.bc import data_loss_factory
    data_loss = data_loss_factory(loss_fn,[0])

    cloud_point_list = [[pde_input, pde_output],[bc_input, bc_output]]
    loss_list = [pde_loss,data_loss]
    return cloud_point_list,loss_list

def get_time_problem(pde_epoch_size = 16, loss_fn = torch.nn.MSELoss()):
    pde_input = D1Field([-1,1]).get_field_rand(pde_epoch_size)
    pde_output = np.zeros([pde_epoch_size,1])

    def pde_loss(model, data):
        x_in,y_real = data
        x_in.requires_grad=True
        U = model(x_in)
        u = U[:,[0]]
        dudx = grad(u, x_in)[0]
        loss = dudx - pi * torch.cos(pi * x_in)
        loss = loss_fn(loss, y_real)
        return loss
    
    ## define a bc
    bc_epoch_size = 2
    bc_input = np.array([[-1]])
    bc_output = np.zeros([bc_epoch_size ,1])

    from user_funn.bc import data_loss_factory
    data_loss = data_loss_factory(loss_fn,[0])

    cloud_point_list = [[pde_input, pde_output],[bc_input, bc_output]]
    loss_list = [pde_loss,data_loss]
    return cloud_point_list,loss_list

def get_long_time_problem(pde_epoch_size = 96, loss_fn = torch.nn.MSELoss()):
    pde_input = np.linspace(0,6,pde_epoch_size).reshape(pde_epoch_size,1)
    pde_output = np.zeros([pde_epoch_size,1])

    def pde_loss(model, data):
        x_in,y_real = data
        x_in.requires_grad=True
        U = model(x_in)
        u = U[:,[0]]
        dudx = grad(u, x_in)[0]
        loss = dudx + pi * torch.sin(pi * x_in)
        loss = loss_fn(loss, y_real)
        return loss
    
    ## define a bc
    bc_input = np.array([[0]])
    bc_output = np.array([[1]])

    from user_funn.bc import data_loss_factory
    data_loss = data_loss_factory(loss_fn,[0])

    cloud_point_list = [[pde_input, pde_output],[bc_input, bc_output]]
    loss_list = [pde_loss,data_loss]
    return cloud_point_list,loss_list