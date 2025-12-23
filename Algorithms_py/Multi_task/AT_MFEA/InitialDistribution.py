import numpy as np


def InitialDistribution(population, no_of_tasks):
    mu_tasks = [None] * no_of_tasks
    Sigma_tasks = [None] * no_of_tasks

    dim = len(population[0].rnvec) if population else 0
    for i in range(no_of_tasks):
        individuals = []
        for ind in population:
            if getattr(ind, 'skill_factor', None) == (i + 1):
                individuals.append(np.asarray(ind.rnvec).ravel())
        if len(individuals) == 0:
            # fallback: zero mean and small identity covariance
            mu_tasks[i] = np.zeros(dim)
            Sigma_tasks[i] = np.eye(dim) * 1e-6
        else:
            arr = np.vstack(individuals)
            mu_tasks[i] = np.mean(arr, axis=0)
            cov = np.cov(arr, rowvar=False)
            # diagonalize
            cov_diag = np.diag(np.diag(cov))
            Sigma_tasks[i] = cov_diag
    return mu_tasks, Sigma_tasks
