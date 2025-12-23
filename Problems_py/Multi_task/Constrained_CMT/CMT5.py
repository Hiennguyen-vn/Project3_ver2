import numpy as np
from Problems_py.Problem import Problem
from Problems_py.Multi_task.Constrained_CMT.Base.ackley1 import ackley1
from Problems_py.Multi_task.Constrained_CMT.Base.rosenbrock2 import rosenbrock2

class CMT5(Problem):
    def __init__(self):
        super().__init__('CMT5')
        self.dims = 50

    def get_parameter(self):
        return self.get_run_parameter() + ['Dims', str(self.dims)]

    def set_parameter(self, parameter_cell):
        self.set_run_parameter(parameter_cell[0:2])
        self.dims = int(parameter_cell[2])
        return self

    def get_tasks(self):
        Tasks = []
        Tasks.append({'dims': self.dims, 'fnc': lambda x: ackley1(x, 1, np.zeros(self.dims), -30 * np.ones(self.dims)), 'Lb': -50 * np.ones(self.dims), 'Ub': 50 * np.ones(self.dims)})
        Tasks.append({'dims': self.dims, 'fnc': lambda x: rosenbrock2(x, 1, np.zeros(self.dims), np.zeros(self.dims)), 'Lb': -50 * np.ones(self.dims), 'Ub': 50 * np.ones(self.dims)})
        return Tasks
