from abc import ABC, abstractmethod

class Problem(ABC):
    """Problem base class translated from MATLAB Problem.m

    Attributes:
        name: problem name
        sub_pop: each task population size
        sub_eva: each task max evaluations
    """
    def __init__(self, name):
        self.name = name
        self.sub_pop = 50
        self.sub_eva = 50 * 1000

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        return self

    def get_run_parameter(self):
        return ['N: Each Task Population Size', str(self.sub_pop), 'E: Each Task Evaluation Max', str(self.sub_eva)]

    def get_run_parameter_list(self):
        return [self.sub_pop, self.sub_eva]

    def set_run_parameter(self, run_parameter):
        self.sub_pop = int(run_parameter[0])
        self.sub_eva = int(run_parameter[1])
        return self

    @abstractmethod
    def get_parameter(self):
        pass

    @abstractmethod
    def set_parameter(self, parameter_cell):
        pass

    @abstractmethod
    def get_tasks(self):
        pass
