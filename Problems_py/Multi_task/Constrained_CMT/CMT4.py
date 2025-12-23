import numpy as np
from Problems_py.Problem import Problem
from Problems_py.Multi_task.Constrained_CMT.Base.rastrigin1 import rastrigin1
from Problems_py.Multi_task.Constrained_CMT.Base.sphere1 import sphere1

class CMT4(Problem):
    def __init__(self):
        super().__init__('CMT4')
        self.dims = 50

    def get_parameter(self):
        return self.get_run_parameter() + ['Dims', str(self.dims)]

    def set_parameter(self, parameter_cell):
        self.set_run_parameter(parameter_cell[0:2])
        self.dims = int(parameter_cell[2])
        return self

    def get_tasks(self):
        Tasks = []
        t1 = {'dims': self.dims, 'fnc': lambda x: rastrigin1(x, 1, np.zeros(self.dims), -20 * np.ones(self.dims)), 'Lb': -50 * np.ones(self.dims), 'Ub': 50 * np.ones(self.dims)}
        Tasks.append(t1)
        t2 = {'dims': self.dims, 'fnc': lambda x: sphere1(x, 1, np.zeros(self.dims), 30 * np.ones(self.dims)), 'Lb': -100 * np.ones(self.dims), 'Ub': 100 * np.ones(self.dims)}
        Tasks.append(t2)
        return Tasks
