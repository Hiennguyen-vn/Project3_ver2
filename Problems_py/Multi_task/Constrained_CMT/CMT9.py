import numpy as np
from Problems_py.Problem import Problem
from Problems_py.Multi_task.Constrained_CMT.Base.rastrigin4 import rastrigin4
from Problems_py.Multi_task.Constrained_CMT.Base.schwefel2 import schwefel2

class CMT9(Problem):
    def __init__(self):
        super().__init__('CMT9')
        self.dims = 50

    def get_parameter(self):
        return self.get_run_parameter() + ['Dims', str(self.dims)]

    def set_parameter(self, parameter_cell):
        self.set_run_parameter(parameter_cell[0:2])
        self.dims = int(parameter_cell[2])
        return self

    def get_tasks(self):
        Tasks = []
        Tasks.append({'dims': self.dims, 'fnc': lambda x: rastrigin4(x, 1, -10 * np.ones(self.dims), np.zeros(self.dims)), 'Lb': -50 * np.ones(self.dims), 'Ub': 50 * np.ones(self.dims)})
        Tasks.append({'dims': self.dims, 'fnc': lambda x: schwefel2(x, 1, np.zeros(self.dims), 100 * np.ones(self.dims)), 'Lb': -500 * np.ones(self.dims), 'Ub': 500 * np.ones(self.dims)})
        return Tasks
