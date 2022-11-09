import numpy as np

def line_sample(begin_point, end_point, sample_size):
    begin_point = np.array(begin_point)
    end_point = np.array(end_point)
    dir_vec =  end_point - begin_point
    r = np.random.rand(sample_size)
    sample_point = begin_point.reshape([1, 2]) + \
        np.array([r,r]).T * dir_vec.reshape([1, 2])
    return sample_point

def line_linspace(begin_point, end_point, sample_size):
    begin_point = np.array(begin_point)
    end_point = np.array(end_point)
    dir_vec = end_point - begin_point
    dir_vec_len = np.linalg.norm(dir_vec)
    dir_vec_nor = dir_vec/dir_vec_len

    dir_vec_linspace_len = np.linspace(0, dir_vec_len, sample_size)
    dir_vec_linspace_len = dir_vec_linspace_len.reshape(sample_size,1)
    dir_vec_linspace = dir_vec_nor.reshape(1,2) * dir_vec_linspace_len
    return begin_point.reshape(1,2) + dir_vec_linspace

def add_t(cloud_point, t_array):
    """ 为一个点云数据添加t方向维度
    Args:
        cloud_point: ndarray with shape=(batch_size * spatial_dimension)
        t_array: 1-d ndarray represent the time
    Res:
        new_cloud_point: ndarray with shape=(batch_size * (spatial_dimension+1))
        d0 is the time dimension.
    """
    num_time = t_array.shape[0]
    t_array = t_array.reshape(num_time,1)
    bs = cloud_point.shape[0]

    rep_t_array = np.zeros([num_time*bs,1])

    k = 0
    for i in range(num_time):
        for _ in range(bs):
            rep_t_array[k] = t_array[i]
            k = k+1


    rep_cloud_point = np.tile(cloud_point, (num_time,1))
    new_cloud_point = np.hstack([rep_t_array, rep_cloud_point])
    return new_cloud_point

