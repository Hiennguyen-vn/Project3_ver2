import numpy as np
from Problems_py.Problem import Problem
from Problems_py.Multi_task.Constrained_CMT.Base.rosenbrock1 import rosenbrock1
from Problems_py.Multi_task.Constrained_CMT.Base.rastrigin1 import rastrigin1

class CMT7(Problem):
    def __init__(self):
        super().__init__('CMT7')
        self.dims = 50

    def get_parameter(self):
        return self.get_run_parameter() + ['Dims', str(self.dims)]

    def set_parameter(self, parameter_cell):
        self.set_run_parameter(parameter_cell[0:2])
        self.dims = int(parameter_cell[2])
        return self

    def get_tasks(self):
        Tasks = []
        Tasks.append({'dims': self.dims, 'fnc': lambda x: rosenbrock1(x, 1, -30 * np.ones(self.dims), -35 * np.ones(self.dims)), 'Lb': -50 * np.ones(self.dims), 'Ub': 50 * np.ones(self.dims)})
        Tasks.append({'dims': self.dims, 'fnc': lambda x: rastrigin1(x, 1, 35 * np.ones(self.dims), 40 * np.ones(self.dims)), 'Lb': -50 * np.ones(self.dims), 'Ub': 50 * np.ones(self.dims)})
        return Tasks
