import time
from Algorithms_py.Algorithm import Algorithm
from Algorithms_py.Utils.initializeMF_FP import initializeMF_FP
from Algorithms_py.Utils.OperatorMFEAalpha import OperatorMFEAalpha
from Algorithms_py.Utils.selectMF_FP import selectMF_FP
from Algorithms_py.Utils.uni2real import uni2real
from Algorithms_py.Utils.Individual import Individual

class FP_MFEA(Algorithm):
    def __init__(self, name='FP_MFEA'):
        super().__init__(name)
        self.rmp = 0.3
        self.mu = 2
        self.mum = 5

    def getParameter(self):
        return {'rmp': self.rmp, 'mu': self.mu, 'mum': self.mum}

    def setParameter(self, parameter_cell):
        self.rmp = float(parameter_cell[0])
        self.mu = float(parameter_cell[1])
        self.mum = float(parameter_cell[2])

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

        generation = 1
        while fnceval_calls < eva_num:
            generation += 1
            offspring, calls = OperatorMFEAalpha.generate(1, population, Tasks, self.rmp, self.mu, self.mum)
            fnceval_calls += calls
            population, bestobj, bestCV, bestX, _ = selectMF_FP(population, offspring, Tasks, pop_size, bestobj, bestCV, bestX)
            data['convergence'].append(list(bestobj))
            data['convergence_cv'].append(list(bestCV))
        # mask infeasible
        import numpy as _np
        conv = _np.array(data['convergence'])
        conv_cv = _np.array(data['convergence_cv'])
        conv[conv_cv > 0] = _np.nan
        data['convergence'] = conv.T.tolist()  # tasks x generations
        data['convergence_cv'] = conv_cv.T.tolist()
        data['bestX'] = uni2real(bestX, Tasks)
        data['clock_time'] = time.time() - start
        return data
