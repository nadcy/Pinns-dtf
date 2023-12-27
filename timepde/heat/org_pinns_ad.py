import torch
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
torch.manual_seed(1)

import sys
sys.path.append('./')

from user_fun.solver.cp_solver import CloudPointSolver
from user_fun.geom import line_sample

density = 32
init_input = line_sample([0,0],[1,0],density)
init_output = np.sin(2*np.pi *init_input[:,[0]])

pass
