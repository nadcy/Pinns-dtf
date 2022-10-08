import os
import sys

add_path = os.path.join(sys.path[0],'../')
sys.path.append(add_path)

from user_funn.geom import line_sample
print(line_sample([0,0],[1,1],10))

from user_funn.geom import line_linspace
print(line_linspace([0,0],[1,1],10))