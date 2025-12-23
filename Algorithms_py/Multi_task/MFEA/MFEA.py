import time
from Algorithms_py.Algorithm import Algorithm
from Algorithms_py.Utils.initialize import initialize
from Algorithms_py.Utils.OperatorMFEAalpha import OperatorMFEAalpha
from Algorithms_py.Utils.uni2real import uni2real
from Algorithms_py.Utils.Individual import Individual

class MFEA(Algorithm):
    def __init__(self, name='MFEA'):
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

        # initializeMF equivalent
        population, calls = initialize(Individual, pop_size, Tasks, len(Tasks))
        # compute bestobj initial
        bestobj = []
        for t in range(len(Tasks)):
            costs = [population[i].factorial_costs[t] for i in range(len(population))]
            bestobj.append(min(costs))
        data = {'convergence': [bestobj]}

        fnceval_calls = calls
        generation = 1
        while fnceval_calls < eva_num:
            generation += 1
            offspring, calls = OperatorMFEAalpha.generate(1, population, Tasks, self.rmp, self.mu, self.mum)
            fnceval_calls += calls
            # selection simplified: keep best by scalar fitness
            population = population + offspring
            # update ranks and scalar fitness
            for t in range(len(Tasks)):
                # assign ranks
                costs = [population[i].constraint_violation[t] for i in range(len(population))]
                rank_cv = sorted(range(len(population)), key=lambda i: costs[i])
                for rank_i, ind_i in enumerate(rank_cv, start=1):
                    if population[ind_i].factorial_ranks is None:
                        population[ind_i].factorial_ranks = [0] * len(Tasks)
                    population[ind_i].factorial_ranks[t] = rank_i
            for i in range(len(population)):
                population[i].scalar_fitness = 1.0 / min(population[i].factorial_ranks)
            rank = sorted(range(len(population)), key=lambda i: -population[i].scalar_fitness)
            population = [population[i] for i in rank[:pop_size]]

            # compute bestobj
            bestobj = []
            for t in range(len(Tasks)):
                costs = [population[i].factorial_costs[t] for i in range(len(population))]
                bestobj.append(min(costs))
            data['convergence'].append(bestobj)

        data['convergence'] = list(map(list, zip(*data['convergence'])))
        data['bestX'] = uni2real([p.rnvec for p in population], Tasks)
        data['clock_time'] = time.time() - start
        return data
