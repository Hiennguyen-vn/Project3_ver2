import random
from Algorithms_py.Utils.Individual import Individual
from Algorithms_py.Utils.evaluate import evaluate


def initialize(Individual_class, pop_size, Task, tasks_num):
    population = []
    if tasks_num == 1:
        for i in range(pop_size):
            ind = Individual_class()
            import numpy as _np
            ind.rnvec = _np.random.rand(Task.dims)
            population.append(ind)
        population, calls = evaluate(population, Task, 1)
    else:
        import numpy as _np
        maxdims = max([t.dims for t in Task])
        for i in range(pop_size):
            ind = Individual_class()
            ind.rnvec = _np.random.rand(maxdims)
            ind.skill_factor = 0
            ind.factorial_costs = [float('inf')] * tasks_num
            ind.constraint_violation = [float('inf')] * tasks_num
            population.append(ind)
        calls = 0
        for t in range(tasks_num):
            population, cal = evaluate(population, Task[t], t + 1)
            calls += cal
    return population, calls
