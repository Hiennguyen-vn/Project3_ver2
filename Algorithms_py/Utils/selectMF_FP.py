def selectMF_FP(population, offspring, Tasks, pop_size, bestobj, bestCV, bestX):
    population = population + offspring
    pop_n = len(population)
    feasible_rate = [0] * len(Tasks)
    for t in range(len(Tasks)):
        constraint_violation = [population[i].constraint_violation[t] for i in range(pop_n)]
        idx = [i for i in range(pop_n) if getattr(population[i], 'skill_factor', 0) == (t+1)]
        feasible_rate[t] = (sum(1 for i in idx if constraint_violation[i] <= 0) / len(idx)) if len(idx) > 0 else 0.0
        rank_cv = sorted(range(pop_n), key=lambda i: constraint_violation[i])
        for rank_i, ind_i in enumerate(rank_cv, start=1):
            if population[ind_i].factorial_ranks is None:
                population[ind_i].factorial_ranks = [0] * len(Tasks)
            population[ind_i].factorial_ranks[t] = rank_i
        bestobj_now = population[rank_cv[0]].factorial_costs[t]
        bestCV_now = constraint_violation[rank_cv[0]]
        best_idx = rank_cv[0]
        if bestCV_now <= 0:
            x_idx = [i for i in range(pop_n) if constraint_violation[i] == bestCV_now]
            factorial_costs = [population[i].factorial_costs[t] for i in x_idx]
            sorted_idx = sorted(range(len(factorial_costs)), key=lambda j: factorial_costs[j])
            idx_sorted = [x_idx[j] for j in sorted_idx]
            for rank_i, ind_i in enumerate(idx_sorted, start=1):
                population[ind_i].factorial_ranks[t] = rank_i
            bestobj_now = population[idx_sorted[0]].factorial_costs[t]
            best_idx = idx_sorted[0]
        if bestCV_now <= bestCV[t] and bestobj_now < bestobj[t]:
            bestobj[t] = bestobj_now
            bestCV[t] = bestCV_now
            bestX[t] = population[best_idx].rnvec
    # scalar fitness
    for i in range(len(population)):
        population[i].scalar_fitness = 1.0 / min(population[i].factorial_ranks)
    rank = sorted(range(len(population)), key=lambda i: -population[i].scalar_fitness)
    newpop = [population[i] for i in rank[:pop_size]]
    return newpop, bestobj, bestCV, bestX, feasible_rate
