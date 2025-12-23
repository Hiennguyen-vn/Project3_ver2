import time
from Algorithms_py.Algorithm import Algorithm
from Algorithms_py.Utils.initialize import initialize
from Algorithms_py.Utils.uni2real import uni2real
from Algorithms_py.Utils.OperatorGA import OperatorGA
from Algorithms_py.Utils.Individual import Individual

class GA(Algorithm):
    def __init__(self, name='GA'):
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
        data = {'convergence': [], 'bestX': []}
        for t_idx in range(len(Tasks)):
            Task = Tasks[t_idx]
            fnceval_calls = 0
            population, calls = initialize(Individual, sub_pop, Task, 1)
            fnceval_calls += calls
            bestobj = min([p.factorial_costs[0] for p in population])
            bestX = [p.rnvec for p in population if p.factorial_costs[0] == bestobj][0]
            convergence = [bestobj]
            generation = 1
            while fnceval_calls < sub_eva:
                generation += 1
                offspring, calls = OperatorGA.generate(1, population, Task, self.mu, self.mum)
                fnceval_calls += calls
                population = population + offspring
                population = sorted(population, key=lambda p: p.factorial_costs[0])[:sub_pop]
                bestobj_now = min([p.factorial_costs[0] for p in population])
                if bestobj_now < bestobj:
                    bestobj = bestobj_now
                    bestX = [p.rnvec for p in population if p.factorial_costs[0] == bestobj_now][0]
                convergence.append(bestobj)
            data['convergence'].append(convergence)
            data['bestX'].append(bestX)
        data['bestX'] = uni2real(data['bestX'], Tasks)
        data['clock_time'] = time.time() - start
        return data
