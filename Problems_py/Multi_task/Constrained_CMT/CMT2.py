import numpy as np
from Problems_py.Problem import Problem
from Problems_py.Multi_task.Constrained_CMT.Base.ackley2 import ackley2
from Problems_py.Multi_task.Constrained_CMT.Base.rastrigin2 import rastrigin2

class CMT2(Problem):
    def __init__(self):
        super().__init__('CMT2')
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
        t1 = {}
        t1['dims'] = self.dims
        t1['fnc'] = lambda x: ackley2(x, 1, np.zeros(self.dims), -4 * np.ones(self.dims))
        t1['Lb'] = -50 * np.ones(self.dims)
        t1['Ub'] = 50 * np.ones(self.dims)
        Tasks.append(t1)
        t2 = {}
        t2['dims'] = self.dims
        t2['fnc'] = lambda x: rastrigin2(x, 1, np.zeros(self.dims), 4 * np.ones(self.dims))
        t2['Lb'] = -50 * np.ones(self.dims)
        t2['Ub'] = 50 * np.ones(self.dims)
        Tasks.append(t2)
        return Tasks
