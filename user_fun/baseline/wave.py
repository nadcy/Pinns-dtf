import torch
import numpy as np
from torch import nn
from user_fun.geom import line_linspace,generate_points_in_rectangle
from user_fun.pde import diff
from user_fun import bc

import torch
from torch import nn
from user_fun.geom import line_linspace,generate_points_in_rectangle
from user_fun.pde import diff
from user_fun import bc

def multi_gaussian(x, peaks, sigmas):
    result = 0
    for p, s in zip(peaks, sigmas):
        result += np.exp(-((x-p)**2)/(2*s**2))
    return result

peaks = [0.2, 0.4, 0.7]
sigmas = [0.05, 0.05, 0.05]

def WaveBenchMark(density = 32):
    a = 3
    
    # Initial condition
    init_input = line_linspace([0,0],[0,3],density*2)
    init_output = np.zeros((init_input.shape[0],1))


    left_input = line_linspace([0,0],[1,0],density*3)
    left_output = multi_gaussian(left_input[:,[0]], peaks, sigmas)

    # Simulation interval
    field_input = generate_points_in_rectangle([0,0],[1,3],density*density*3)
    field_output = np.zeros((field_input.shape[0],1))

    cp_list = [
        [field_input, field_output],
        [init_input, init_output],
        [left_input, left_output],
    ]

    loss_fn = nn.MSELoss()
    
    def wave_loss(model, data):
        input,output = data
        input.requires_grad=True
        
        # Data extraction
        t = input[:,[0]]
        x = input[:,[1]]
        use_input = torch.cat([t,x],dim = 1)
        U = model(use_input)
        u = U[:,[0]]

        # Compute first order derivatives
        dudx = diff(u, x)
        dudt = diff(u, t)

        # Loss computation
        loss = dudt + a * dudx
        loss = loss_fn(loss, output)
        return loss

    init_loss = bc.data_loss_factory(loss_fn)
    left_loss = bc.data_loss_factory(loss_fn)
    loss_list = [
        wave_loss, init_loss, left_loss
    ]

    return cp_list, loss_list

cp_list, loss_list = WaveBenchMark()