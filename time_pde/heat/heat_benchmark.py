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
    def __init__(self, n, a, L,tmin = 0, tmax = 1):
        self.n = n
        self.a = a
        self.L = L
        self.tmin = tmin
        self.tmax = tmax

    def heat_eq_exact_solution(self, t, x):
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
                usol[i][j] = self.heat_eq_exact_solution(t[i],x[j])

        return tt,xx,usol

    def gen_testdata(self):
        """Generate test data"""
        tt, xx, exact = self.gen_exact_solution()
        X = np.vstack((np.ravel(tt), np.ravel(xx))).T # use T because meshgrid
        y = exact.flatten()[:, None]
        return X, y
    
    def gen_bc_data(self, x_data):
        pass

    