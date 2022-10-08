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
