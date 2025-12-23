import random
import math

class OperatorGA:
    @staticmethod
    def generate(callfun, population, Task, mu, mum):
        if len(population) <= 2:
            return population[:], 0
        Individual_class = type(population[0])
        indorder = list(range(len(population)))
        random.shuffle(indorder)
        offspring = []
        count = 0
        dim = len(population[0].rnvec)
        for i in range((len(population) + 1) // 2):
            p1 = indorder[i]
            p2 = indorder[(i + len(population) // 2) % len(population)]
            # create two offspring
            o1 = Individual_class()
            o2 = Individual_class()
            import numpy as _np
            u = _np.random.rand(dim)
            cf = _np.zeros(dim)
            cf[u <= 0.5] = (2 * u[u <= 0.5]) ** (1 / (mu + 1))
            cf[u > 0.5] = (2 * (1 - u[u > 0.5])) ** (-1 / (mu + 1))
            # crossover
            o1.rnvec = 0.5 * ((1 + cf) * population[p1].rnvec + (1 - cf) * population[p2].rnvec)
            o2.rnvec = 0.5 * ((1 + cf) * population[p2].rnvec + (1 - cf) * population[p1].rnvec)
            # mutate
            for o in (o1, o2):
                rn = o.rnvec.copy()
                for j in range(dim):
                    if random.random() < 1.0 / dim:
                        u = random.random()
                        if u <= 0.5:
                            delta = (2 * u) ** (1 / (1 + mum)) - 1
                            rn[j] = o.rnvec[j] + delta * o.rnvec[j]
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1 / (1 + mum))
                            rn[j] = o.rnvec[j] + delta * (1 - o.rnvec[j])
                rn[rn > 1] = 1
                rn[rn < 0] = 0
                o.rnvec = rn
            offspring.extend([o1, o2])
            count += 2
        # evaluate
        if callfun:
            from Algorithms_py.Utils.evaluate import evaluate
            offspring, calls = evaluate(offspring, Task, 1)
        else:
            calls = 0
        return offspring, calls
