import pandas as pd
import numpy as np
def read2D_paraview_csv(csv_name, time_val):
    T = pd.read_csv(csv_name)
    T = T[T["Time"]==time_val]
    T = T[T["Points:2"] == 0]
    x = T["Points:0"].to_numpy()
    y = T["Points:1"].to_numpy()
    p = T["p"].to_numpy()
    u = T["U:0"].to_numpy()
    v = T["U:1"].to_numpy()
    return x,y,p,u,v