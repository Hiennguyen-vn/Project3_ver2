import time
from Algorithms_py.Algorithm import Algorithm
from Algorithms_py.Utils.initializeMF_FP import initializeMF_FP
from Algorithms_py.Multi_task.AT_MFEA.OperatorMFEA_AT import OperatorMFEA_AT
from Algorithms_py.Utils.selectMF_FP import selectMF_FP
from Algorithms_py.Utils.uni2real import uni2real
from Algorithms_py.Utils.Individual import Individual


class FP_AT_MFEA(Algorithm):
    def __init__(self, name='FP_AT_MFEA'):
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

        population, fnceval_calls, bestobj, bestCV, bestX, _ = initializeMF_FP(Individual, pop_size, Tasks, len(Tasks))
        data = {}
        data['convergence'] = [bestobj]
        data['convergence_cv'] = [bestCV]
        # initialize affine transformation
        from Algorithms_py.Multi_task.AT_MFEA.InitialDistribution import InitialDistribution
        from Algorithms_py.Multi_task.AT_MFEA.DistributionUpdate import DistributionUpdate
        mu_tasks, Sigma_tasks = InitialDistribution(population, len(Tasks))

        generation = 1
        while fnceval_calls < eva_num:
            generation += 1
            offspring, calls = OperatorMFEA_AT.generate(1, population, Tasks, self.rmp, self.mu, self.mum, self.probswap, mu_tasks, Sigma_tasks)
            fnceval_calls += calls
            population, bestobj, bestCV, bestX, _ = selectMF_FP(population, offspring, Tasks, pop_size, bestobj, bestCV, bestX)
            data['convergence'].append(list(bestobj))
            data['convergence_cv'].append(list(bestCV))
            mu_tasks, Sigma_tasks = DistributionUpdate(mu_tasks, Sigma_tasks, population, len(Tasks))

        import numpy as _np
        conv = _np.array(data['convergence'])
        conv_cv = _np.array(data['convergence_cv'])
        conv[conv_cv > 0] = _np.nan
        data['convergence'] = conv.T.tolist()
        data['convergence_cv'] = conv_cv.T.tolist()
        data['bestX'] = uni2real(bestX, Tasks)
        data['clock_time'] = time.time() - start
        return data
