import random

class OperatorMFEAalpha:
    @staticmethod
    def generate(callfun, population, Tasks, rmp, mu, mum):
        if not population:
            return [], 0
        Individual_class = type(population[0])
        import numpy as _np
        indorder = list(range(len(population)))
        _np.random.shuffle(indorder)
        offspring = []
        dim = max([t.dims for t in Tasks])
        for i in range((len(population) + 1) // 2):
            p1 = indorder[i]
            p2 = indorder[(i + len(population) // 2) % len(population)]
            o1 = Individual_class()
            o1.factorial_costs = [float('inf')] * len(Tasks)
            o1.constraint_violation = [float('inf')] * len(Tasks)
            o2 = Individual_class()
            o2.factorial_costs = [float('inf')] * len(Tasks)
            o2.constraint_violation = [float('inf')] * len(Tasks)

            u = _np.random.rand(dim)
            cf = _np.zeros(dim)
            cf[u <= 0.5] = (2 * u[u <= 0.5]) ** (1 / (mu + 1))
            cf[u > 0.5] = (2 * (1 - u[u > 0.5])) ** (-1 / (mu + 1))

            if (getattr(population[p1], 'skill_factor', None) == getattr(population[p2], 'skill_factor', None)) or random.random() < rmp:
                # crossover + mutate
                o1.rnvec = 0.5 * ((1 + cf) * population[p1].rnvec + (1 - cf) * population[p2].rnvec)
                o2.rnvec = 0.5 * ((1 + cf) * population[p2].rnvec + (1 - cf) * population[p1].rnvec)
                # mutation
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
                # imitate skill factor
                o1.skill_factor = population[random.choice([p1, p2])].skill_factor
                o2.skill_factor = population[random.choice([p1, p2])].skill_factor
            else:
                # mutate parents and copy skill factor
                for idx, p in enumerate([p1, p2]):
                    new = Individual_class()
                    rn = population[p].rnvec.copy()
                    for j in range(dim):
                        if random.random() < 1.0 / dim:
                            u = random.random()
                            if u <= 0.5:
                                delta = (2 * u) ** (1 / (1 + mum)) - 1
                                rn[j] = population[p].rnvec[j] + delta * population[p].rnvec[j]
                            else:
                                delta = 1 - (2 * (1 - u)) ** (1 / (1 + mum))
                                rn[j] = population[p].rnvec[j] + delta * (1 - population[p].rnvec[j])
                    rn[rn > 1] = 1
                    rn[rn < 0] = 0
                    new.rnvec = rn
                    new.skill_factor = population[p].skill_factor
                    if idx == 0:
                        o1 = new
                    else:
                        o2 = new
            offspring.extend([o1, o2])
        # evaluate offspring grouped by skill_factor
        if callfun:
            from Algorithms_py.Utils.evaluate import evaluate
            offspring_temp = []
            calls = 0
            for t_idx in range(len(Tasks)):
                group = [o for o in offspring if getattr(o, 'skill_factor', None) == (t_idx + 1)]
                if group:
                    group, cal = evaluate(group, Tasks[t_idx], t_idx + 1)
                    offspring_temp.extend(group)
                    calls += cal
            offspring = offspring_temp
        else:
            calls = 0
        return offspring, calls
