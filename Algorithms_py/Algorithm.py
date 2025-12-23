from abc import ABC, abstractmethod

class Algorithm(ABC):
    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    @abstractmethod
    def getParameter(self):
        pass

    @abstractmethod
    def setParameter(self, parameter_cell):
        pass

    @abstractmethod
    def run(self, Tasks, run_parameter_list):
        pass
