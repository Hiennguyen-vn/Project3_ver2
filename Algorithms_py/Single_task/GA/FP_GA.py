import time
from Algorithms_py.Algorithm import Algorithm
from Algorithms_py.Utils.initialize import initialize
from Algorithms_py.Utils.uni2real import uni2real
from Algorithms_py.Utils.OperatorGA import OperatorGA
from Algorithms_py.Utils.Individual import Individual

class FP_GA(Algorithm):
    def __init__(self, name='FP_GA'):
        super().__init__(name)
        self.mu = 2
        self.mum = 5

    def getParameter(self):
        return {'mu': self.mu, 'mum': self.mum}

    def setParameter(self, parameter_cell):
        self.mu = float(parameter_cell[0])
        self.mum = float(parameter_cell[1])

    def run(self, Tasks, run_parameter_list):
        sub_pop = run_parameter_list[0]
        sub_eva = run_parameter_list[1]
        start = time.time()
        data = {'convergence': [], 'convergence_cv': [], 'bestX': []}
        for t_idx in range(len(Tasks)):
            Task = Tasks[t_idx]
            fnceval_calls = 0
            population, calls = initialize(Individual, sub_pop, Task, 1)
            fnceval_calls += calls
            # select best by constraint violation then cost
            bestCV = min([pop.constraint_violation[0] for pop in population])
            pop_temp = [p for p in population if p.constraint_violation[0] == bestCV]
            bestobj = min([p.factorial_costs[0] for p in pop_temp])
            best_idx = [i for i,p in enumerate(pop_temp) if p.factorial_costs[0] == bestobj][0]
            bestX = pop_temp[best_idx].rnvec
            convergence = [bestobj]
            convergence_cv = [bestCV]
            generation = 1
            while fnceval_calls < sub_eva:
                generation += 1
                offspring, calls = OperatorGA.generate(1, population, Task, self.mu, self.mum)
                fnceval_calls += calls
                population = population + offspring
                feasible_num = sum(1 for p in population if p.constraint_violation[0] == 0)
                if feasible_num < sub_pop:
                    population = sorted(population, key=lambda p: p.constraint_violation[0])[:sub_pop]
                else:
                    pop_temp = [p for p in population if p.constraint_violation[0] == 0]
                    population = sorted(pop_temp, key=lambda p: p.factorial_costs[0])[:sub_pop]
                bestCV_now = min([p.constraint_violation[0] for p in population])
                pop_temp = [p for p in population if p.constraint_violation[0] == bestCV_now]
                bestobj_now = min([p.factorial_costs[0] for p in pop_temp])
                if bestCV_now <= bestCV and bestobj_now < bestobj:
                    bestobj = bestobj_now
                    bestCV = bestCV_now
                    bestX = [p for p in pop_temp if p.factorial_costs[0] == bestobj_now][0].rnvec
                convergence.append(bestobj)
                convergence_cv.append(bestCV)
            # store
            # convert to arrays/lists
            data['convergence'].append(convergence)
            data['convergence_cv'].append(convergence_cv)
            data['bestX'].append(bestX)
        # mask infeasible
        import numpy as _np
        conv = _np.array([_np.pad(_np.array(c), (0, max(0, max(len(r) for r in data['convergence']) - len(c))), constant_values=_np.nan) for c in data['convergence']])
        conv_cv = _np.array([_np.pad(_np.array(c), (0, max(0, max(len(r) for r in data['convergence_cv']) - len(c))), constant_values=_np.nan) for c in data['convergence_cv']])
        conv[conv_cv > 0] = _np.nan
        data['convergence'] = conv.tolist()
        data['convergence_cv'] = conv_cv.tolist()
        data['bestX'] = uni2real(data['bestX'], Tasks)
        data['clock_time'] = time.time() - start
        return data
