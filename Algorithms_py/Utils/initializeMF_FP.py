from Algorithms_py.Utils.initialize import initialize


def initializeMF_FP(Individual_class, pop_size, Tasks, tasks_num):
    population, calls = initialize(Individual_class, pop_size, Tasks, tasks_num)
    feasible_rate = []
    bestobj = [None] * len(Tasks)
    bestCV = [None] * len(Tasks)
    bestX = [None] * len(Tasks)

    for t in range(len(Tasks)):
        constraint_violation = [population[i].constraint_violation[t] for i in range(pop_size)]
        # idx: those with skill_factor == t+1 (MATLAB indices start at 1)
        idx = [i for i in range(pop_size) if getattr(population[i], 'skill_factor', 0) == (t + 1)]
        feasible_rate_t = None
        if len(idx) > 0:
            feasible_rate_t = sum(1 for i in idx if constraint_violation[i] <= 0) / len(idx)
        else:
            feasible_rate_t = 0.0
        feasible_rate.append(feasible_rate_t)
        bestCV[t] = min(constraint_violation)
        rank_cv = sorted(range(pop_size), key=lambda i: constraint_violation[i])
        for rank_i, ind_i in enumerate(rank_cv, start=1):
            if population[ind_i].factorial_ranks is None:
                population[ind_i].factorial_ranks = [0] * len(Tasks)
            population[ind_i].factorial_ranks[t] = rank_i
        bestobj[t] = population[rank_cv[0]].factorial_costs[t]
        bestX[t] = population[rank_cv[0]].rnvec
        if bestCV[t] <= 0:
            x_idx = [i for i in range(pop_size) if constraint_violation[i] == bestCV[t]]
            factorial_costs = [population[i].factorial_costs[t] for i in x_idx]
            sorted_idx = sorted(range(len(factorial_costs)), key=lambda j: factorial_costs[j])
            idx_sorted = [x_idx[j] for j in sorted_idx]
            for rank_i, ind_i in enumerate(idx_sorted, start=1):
                population[ind_i].factorial_ranks[t] = rank_i
            bestobj[t] = population[idx_sorted[0]].factorial_costs[t]
            bestX[t] = population[idx_sorted[0]].rnvec

    # Calculate skill factor
    import random
    for i in range(pop_size):
        min_rank = min(population[i].factorial_ranks)
        min_idx = [j for j, v in enumerate(population[i].factorial_ranks) if v == min_rank]
        # choose random among ties
        population[i].skill_factor = random.choice([j + 1 for j in min_idx])
        # set factorial_costs to inf except skill factor
        for k in range(population[i].skill_factor - 1):
            population[i].factorial_costs[k] = float('inf')
        for k in range(population[i].skill_factor, len(Tasks)):
            population[i].factorial_costs[k] = float('inf')

    return population, calls, bestobj, bestCV, bestX, feasible_rate
