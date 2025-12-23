import numpy as np
from Problems_py.Problem import Problem
from Problems_py.Multi_task.Constrained_CMT.Base.griewank1 import griewank1
from Problems_py.Multi_task.Constrained_CMT.Base.rastrigin1 import rastrigin1

class CMT1(Problem):
    def __init__(self):
        super().__init__('CMT1')
        self.dims = 50

    def get_parameter(self):
        parameter = ['Dims', str(self.dims)]
        return self.get_run_parameter() + parameter

    def set_parameter(self, parameter_cell):
        self.set_run_parameter(parameter_cell[0:2])
        self.dims = int(parameter_cell[2])
        return self

    def get_tasks(self):
        Tasks = []
        # Task 1
        task1 = {}
        task1['dims'] = self.dims
        task1['fnc'] = lambda x: griewank1(x, 1, np.zeros(self.dims), -40 * np.ones(self.dims))
        task1['Lb'] = -100 * np.ones(self.dims)
        task1['Ub'] = 100 * np.ones(self.dims)
        Tasks.append(task1)
        # Task 2
        task2 = {}
        task2['dims'] = self.dims
        task2['fnc'] = lambda x: rastrigin1(x, 1, np.zeros(self.dims), 20 * np.ones(self.dims))
        task2['Lb'] = -50 * np.ones(self.dims)
        task2['Ub'] = 50 * np.ones(self.dims)
        Tasks.append(task2)
        return Tasks
