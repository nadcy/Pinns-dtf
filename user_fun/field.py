import numpy as np

class Field:
    # field_lower,field_upper should be a numpy array
    def __init__(self, field_lower, field_upper):
        self.field_lower = field_lower
        self.field_upper = field_upper
        self.field_range = field_upper - field_lower
    def get_field_rand(self,data_size):
        x_data = np.random.rand(data_size,2)
        x_data = x_data * self.field_range.reshape(1,2) +self.field_lower.reshape(1,2)
        return x_data
    def get_field_mesh(self,use_point_per_dim = [10,10],DX = 0):
        x0 = np.linspace(self.field_lower[0]+DX,
            self.field_upper[0]-DX,use_point_per_dim[0])
        x1 = np.linspace(self.field_lower[1]+DX,
            self.field_upper[1]-DX,use_point_per_dim[1])
        X0,X1 = np.meshgrid(x0,x1)
        X = np.array([X0.ravel(),X1.ravel()]).T
        return X

class D1Field:
    # field_lower,field_upper should be a numpy array
    def __init__(self, x_range):
        self.x_range = np.array(x_range)
    
    def get_field_rand(self,data_size):
        # 范围处理
        x_lower_range = np.array([self.x_range[0]])
        x_upper_range = np.array([self.x_range[1]])
        x_range = x_upper_range - x_lower_range

        # 随机数据生成
        x_data = np.random.rand(data_size,1)
        x_data = x_data * x_range + x_lower_range
        return x_data

    def get_field_mesh(self, use_point_per_dim = [10,10], margin = 0):
        x0_point_vec = np.linspace(self.x0_range[0],self.x0_range[1],use_point_per_dim[0])
        x1_point_vec = np.linspace(self.x1_range[0],self.x1_range[1],use_point_per_dim[1])
        X0,X1 = np.mgrid(x0_point_vec,x1_point_vec)
        X = np.array([X0.ravel(),X1.ravel()]).T
        return X



class D2Field:
    # field_lower,field_upper should be a numpy array
    def __init__(self, x0_range, x1_range):
        self.x0_range = np.array(x0_range)
        self.x1_range = np.array(x1_range)
    
    def get_field_rand(self,data_size):
        # 范围处理
        x_lower_range = np.array([self.x0_range[0],self.x1_range[0]])
        x_upper_range = np.array([self.x0_range[1],self.x1_range[1]])
        x_range = x_upper_range - x_lower_range
        # 随机数据生成
        x_data = np.random.rand(data_size,2)
        x_data = x_data * x_range.reshape(1,2) + x_lower_range.reshape(1,2)
        return x_data

    def get_field_mesh(self, use_point_per_dim = [10,10], margin = 0):
        x0_point_vec = np.linspace(self.x0_range[0],self.x0_range[1],use_point_per_dim[0])
        x1_point_vec = np.linspace(self.x1_range[0],self.x1_range[1],use_point_per_dim[1])
        X0,X1 = np.mgrid(x0_point_vec,x1_point_vec)
        X = np.array([X0.ravel(),X1.ravel()]).T
        return X

    def define_input_variable(self, variable_str = ['x','y']):
        self.input_variable_name = variable_str

    def define_output_variable(self, variable_str):
        self.output_variable_name = variable_str
    
    ######################################################
    # Example: get_bc_rand(['R', 0], lambda p,u,v:0,0,0)
    # Meaning：when y=0, p=0,u=0,v=0
    ######################################################
    def get_bc_rand(self, data_size, bc_describe_str, output_fun):
        # generate x_data
        x_data = self.get_field_rand(data_size)

        for i in range(3):
            if bc_describe_str[i] == 'R':
                continue
            else:
                x_data[:,i] = bc_describe_str[i]
        
        # generate y_data
        y_data = []
        for x_data_item in x_data:
            y_data.append(output_fun(*x_data_item))
        y_data = np.array(y_data)
        return x_data,y_data


class D2t1_field:
    # field_lower,field_upper should be a numpy array
    def __init__(self, x0_range, x1_range, t_range):
        self.x0_range = np.array(x0_range)
        self.x1_range = np.array(x1_range)
        self.t_range = np.array(t_range)
    
    def get_field_rand(self,data_size):
        # 范围处理
        x_lower_range = np.array([self.x0_range[0],self.x1_range[0],self.t_range[0]])
        x_upper_range = np.array([self.x0_range[1],self.x1_range[1],self.t_range[1]])
        x_range = x_upper_range - x_lower_range
        # 随机数据生成
        x_data = np.random.rand(data_size,3)
        x_data = x_data * x_range.reshape(1,3) + x_lower_range.reshape(1,3)
        return x_data

    def get_field_mesh(self, use_point_per_dim = [10,10,10], margin = 0):
        x0_point_vec = np.linspace(self.x0_range[0],self.x0_range[1],use_point_per_dim[0])
        x1_point_vec = np.linspace(self.x1_range[0],self.x1_range[1],use_point_per_dim[1])
        t_point_vec = np.linspace(self.t_range[0],self.t_range[1],use_point_per_dim[2])
        X0,X1,T = np.mgrid(x0_point_vec,x1_point_vec,t_point_vec)
        X = np.array([X0.ravel(),X1.ravel()],T).T
        return X

    def define_input_variable(self, variable_str = ['x','y','t']):
        self.input_variable_name = variable_str

    def define_output_variable(self, variable_str):
        self.output_variable_name = variable_str
    
    ######################################################
    # Example: get_bc_rand(['R','R', 0], lambda x,y,t:0,0,0)
    # Meaning：when t=0, p=0,u=0,v=0
    ######################################################
    def get_bc_rand(self, data_size, bc_describe_str, output_fun):
        # generate x_data
        x_data = self.get_field_rand(data_size)

        for i in range(3):
            if bc_describe_str[i] == 'R':
                continue
            else:
                x_data[:,i] = bc_describe_str[i]
        
        # generate y_data
        y_data = []
        for x_data_item in x_data:
            y_data.append(output_fun(*x_data_item))
        y_data = np.array(y_data)
        return x_data,y_data
