import time
from Algorithms_py.Algorithm import Algorithm
from Algorithms_py.Utils.initializeMF_FP import initializeMF_FP as initializeMF
from Algorithms_py.Multi_task.AT_MFEA.InitialDistribution import InitialDistribution
from Algorithms_py.Multi_task.AT_MFEA.DistributionUpdate import DistributionUpdate
from Algorithms_py.Multi_task.AT_MFEA.OperatorMFEA_AT import OperatorMFEA_AT
from Algorithms_py.Utils.selectMF_FP import selectMF_FP as selectMF
from Algorithms_py.Utils.uni2real import uni2real
from Algorithms_py.Utils.Individual import Individual


class AT_MFEA(Algorithm):
    def __init__(self, name='AT_MFEA'):
        super().__init__(name)
        self.rmp = 0.3
        self.mu = 2
        self.mum = 5
        self.probswap = 0

    def getParameter(self):
        return {'rmp': self.rmp, 'mu': self.mu, 'mum': self.mum, 'probswap': self.probswap}

    def setParameter(self, parameter_cell):
        self.rmp = float(parameter_cell[0])
        self.mu = float(parameter_cell[1])
        self.mum = float(parameter_cell[2])
        self.probswap = float(parameter_cell[3])

    def run(self, Tasks, run_parameter_list):
        sub_pop = run_parameter_list[0]
        sub_eva = run_parameter_list[1]
        pop_size = int(sub_pop * len(Tasks))
        eva_num = int(sub_eva * len(Tasks))
        start = time.time()

        population, fnceval_calls, bestobj, bestX = initializeMF(Individual, pop_size, Tasks, len(Tasks))
        data = {}
        data['convergence'] = [bestobj]
        # initialize affine transformation
        mu_tasks, Sigma_tasks = InitialDistribution(population, len(Tasks))

        generation = 1
        while fnceval_calls < eva_num:
            generation += 1
            offspring, calls = OperatorMFEA_AT.generate(1, population, Tasks, self.rmp, self.mu, self.mum, self.probswap, mu_tasks, Sigma_tasks)
            fnceval_calls += calls
            population, bestobj, bestX = selectMF(population, offspring, Tasks, pop_size, bestobj, bestX)
            data['convergence'].append(list(bestobj))
            mu_tasks, Sigma_tasks = DistributionUpdate(mu_tasks, Sigma_tasks, population, len(Tasks))

        data['bestX'] = uni2real(bestX, Tasks)
        data['clock_time'] = time.time() - start
        return data
