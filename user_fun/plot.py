from cmath import inf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def scatter_2d_cloud_point_kind(cloud_point_list):
    """
    cloud_point_list:
        a list of cloud point pair like [x_data1, x_data2, ..., x_datan].
    """
    fig,ax = plt.subplots()
    for x_data in cloud_point_list:
        ax.scatter(x_data[:,0].ravel(), x_data[:,1].ravel())


def scatter_2d_cloud_point_pair(cloud_point_list):
    """
    Args:
    cloud_point_list:
        a list of cloud point pair like [(x_data1,y_data1),(x_data2,y_data2)...
        (x_datan,y_datan)].
    
    """
    val_min = np.inf
    val_max = -np.inf
    for cloud_point_item in cloud_point_list:
        _,y_data = cloud_point_item
        val_min = min(np.min(y_data[:,0]), val_min)
        val_max = max(np.max(y_data[:,0]), val_max)

    cmap = matplotlib.colormaps['viridis']
    norm = matplotlib.colors.Normalize(vmin=val_min, vmax=val_max)
    
    fig,ax = plt.subplots()
    for cloud_point_item in cloud_point_list:
        x_data,y_data = cloud_point_item
        cm1 = ax.scatter(x_data[:,0].ravel(),x_data[:,1].ravel(),
            c=y_data[:,0].ravel(),
            cmap = cmap,norm = norm)
    
    fig.colorbar(cm1)