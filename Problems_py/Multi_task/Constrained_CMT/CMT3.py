import numpy as np
from Problems_py.Problem import Problem
from Problems_py.Multi_task.Constrained_CMT.Base.ackley2 import ackley2
from Problems_py.Multi_task.Constrained_CMT.Base.schwefel1 import schwefel1

class CMT3(Problem):
    def __init__(self):
        super().__init__('CMT3')
        self.dims = 50

    def get_parameter(self):
        return self.get_run_parameter() + ['Dims', str(self.dims)]

    def set_parameter(self, parameter_cell):
        self.set_run_parameter(parameter_cell[0:2])
        self.dims = int(parameter_cell[2])
        return self

    def get_tasks(self):
        Tasks = []
        t1 = {}
        t1['dims'] = self.dims
        t1['fnc'] = lambda x: ackley2(x, 1, 42.096 * np.ones(self.dims), 40 * np.ones(self.dims))
        t1['Lb'] = -50 * np.ones(self.dims)
        t1['Ub'] = 50 * np.ones(self.dims)
        Tasks.append(t1)
        t2 = {}
        t2['dims'] = self.dims
        t2['fnc'] = lambda x: schwefel1(x, 1, np.zeros(self.dims), 400 * np.ones(self.dims))
        t2['Lb'] = -500 * np.ones(self.dims)
        t2['Ub'] = 500 * np.ones(self.dims)
        Tasks.append(t2)
        return Tasks
