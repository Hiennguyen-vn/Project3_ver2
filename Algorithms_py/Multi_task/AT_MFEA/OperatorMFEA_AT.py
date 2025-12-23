import random
import numpy as np
from Algorithms_py.Utils.OperatorGA import OperatorGA
from Algorithms_py.Utils.evaluate import evaluate
from Algorithms_py.Multi_task.AT_MFEA.AT_Transfer import AT_Transfer


class OperatorMFEA_AT:
    @staticmethod
    def generate(callfun, population, Tasks, rmp, mu, mum, probswap, mu_tasks, Sigma_tasks):
        if not population:
            return [], 0
        Individual_class = type(population[0])
        indorder = list(range(len(population)))
        np.random.shuffle(indorder)
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

            u = np.random.rand(dim)
            cf = np.zeros(dim)
            cf[u <= 0.5] = (2 * u[u <= 0.5]) ** (1 / (mu + 1))
            cf[u > 0.5] = (2 * (1 - u[u > 0.5])) ** (-1 / (mu + 1))

            sf1 = getattr(population[p1], 'skill_factor', None)
            sf2 = getattr(population[p2], 'skill_factor', None)

            if sf1 == sf2:
                # crossover
                o1.rnvec = 0.5 * ((1 + cf) * population[p1].rnvec + (1 - cf) * population[p2].rnvec)
                o2.rnvec = 0.5 * ((1 + cf) * population[p2].rnvec + (1 - cf) * population[p1].rnvec)
                # mutate
                for o in (o1, o2):
                    rn = o.rnvec.copy()
                    for j in range(dim):
                        if random.random() < 1.0 / dim:
                            uu = random.random()
                            if uu <= 0.5:
                                delta = (2 * uu) ** (1 / (1 + mum)) - 1
                                rn[j] = o.rnvec[j] + delta * o.rnvec[j]
                            else:
                                delta = 1 - (2 * (1 - uu)) ** (1 / (1 + mum))
                                rn[j] = o.rnvec[j] + delta * (1 - o.rnvec[j])
                    rn[rn > 1] = 1
                    rn[rn < 0] = 0
                    o.rnvec = rn
                # variable swap
                swap_indicator = (np.random.rand(dim) >= probswap)
                temp = o2.rnvec[swap_indicator].copy()
                o2.rnvec[swap_indicator] = o1.rnvec[swap_indicator]
                o1.rnvec[swap_indicator] = temp
                # imitate
                o1.skill_factor = population[random.choice([p1, p2])].skill_factor
                o2.skill_factor = population[random.choice([p1, p2])].skill_factor
            elif random.random() < rmp:
                # affine transformation
                pm1 = population[p1]
                pm2 = population[p2]
                pm1_rn = AT_Transfer(population[p1].rnvec, mu_tasks[population[p1].skill_factor - 1], Sigma_tasks[population[p1].skill_factor - 1], mu_tasks[population[p2].skill_factor - 1], Sigma_tasks[population[p2].skill_factor - 1])
                pm2_rn = AT_Transfer(population[p2].rnvec, mu_tasks[population[p2].skill_factor - 1], Sigma_tasks[population[p2].skill_factor - 1], mu_tasks[population[p1].skill_factor - 1], Sigma_tasks[population[p1].skill_factor - 1])
                pm1 = population[p1]
                pm2 = population[p2]
                pm1 = type(pm1)()
                pm2 = type(pm2)()
                pm1.rnvec = pm1_rn
                pm2.rnvec = pm2_rn
                # crossover
                o1.rnvec = 0.5 * ((1 + cf) * pm1.rnvec + (1 - cf) * population[p2].rnvec)
                o2.rnvec = 0.5 * ((1 + cf) * population[p1].rnvec + (1 - cf) * pm2.rnvec)
                # mutate
                for o in (o1, o2):
                    rn = o.rnvec.copy()
                    for j in range(dim):
                        if random.random() < 1.0 / dim:
                            uu = random.random()
                            if uu <= 0.5:
                                delta = (2 * uu) ** (1 / (1 + mum)) - 1
                                rn[j] = o.rnvec[j] + delta * o.rnvec[j]
                            else:
                                delta = 1 - (2 * (1 - uu)) ** (1 / (1 + mum))
                                rn[j] = o.rnvec[j] + delta * (1 - o.rnvec[j])
                    rn[rn > 1] = 1
                    rn[rn < 0] = 0
                    o.rnvec = rn
                o1.skill_factor = population[random.choice([p1, p2])].skill_factor
                o2.skill_factor = population[random.choice([p1, p2])].skill_factor
            else:
                # Randomly pick another individual from the same task
                for x in range(2):
                    p = [p1, p2]
                    find_idx = [idx for idx, val in enumerate(population) if getattr(val, 'skill_factor', None) == population[p[x]].skill_factor]
                    idx = random.choice(find_idx)
                    while idx == p[x] and len(find_idx) > 1:
                        idx = random.choice(find_idx)
                    temp_offspring = Individual_class()
                    # crossover
                    if x == 0:
                        o1.rnvec = 0.5 * ((1 + cf) * population[p1].rnvec + (1 - cf) * population[idx].rnvec)
                        temp_offspring.rnvec = 0.5 * ((1 + cf) * population[idx].rnvec + (1 - cf) * population[p1].rnvec)
                    else:
                        o2.rnvec = 0.5 * ((1 + cf) * population[p2].rnvec + (1 - cf) * population[idx].rnvec)
                        temp_offspring.rnvec = 0.5 * ((1 + cf) * population[idx].rnvec + (1 - cf) * population[p2].rnvec)
                    # mutate
                    for target in (o1 if x == 0 else o2, temp_offspring):
                        rn = target.rnvec.copy()
                        for j in range(dim):
                            if random.random() < 1.0 / dim:
                                uu = random.random()
                                if uu <= 0.5:
                                    delta = (2 * uu) ** (1 / (1 + mum)) - 1
                                    rn[j] = target.rnvec[j] + delta * target.rnvec[j]
                                else:
                                    delta = 1 - (2 * (1 - uu)) ** (1 / (1 + mum))
                                    rn[j] = target.rnvec[j] + delta * (1 - target.rnvec[j])
                        rn[rn > 1] = 1
                        rn[rn < 0] = 0
                        target.rnvec = rn
                    # variable swap if needed
                    swap_indicator = (np.random.rand(dim) >= probswap)
                    if x == 0:
                        o1.rnvec[swap_indicator] = temp_offspring.rnvec[swap_indicator]
                        o1.skill_factor = population[p1].skill_factor
                    else:
                        o2.rnvec[swap_indicator] = temp_offspring.rnvec[swap_indicator]
                        o2.skill_factor = population[p2].skill_factor

            # clip
            for o in (o1, o2):
                o.rnvec[o.rnvec > 1] = 1
                o.rnvec[o.rnvec < 0] = 0
            offspring.extend([o1, o2])

        # evaluate offspring grouped by skill_factor
        if callfun:
            offspring_temp = []
            calls = 0
            for t in range(len(Tasks)):
                group = [o for o in offspring if getattr(o, 'skill_factor', None) == (t + 1)]
                if group:
                    group, cal = evaluate(group, Tasks[t], t + 1)
                    offspring_temp.extend(group)
                    calls += cal
            offspring = offspring_temp
        else:
            calls = 0
        return offspring, calls
