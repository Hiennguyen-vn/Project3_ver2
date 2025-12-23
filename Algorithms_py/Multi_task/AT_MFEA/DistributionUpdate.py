import numpy as np


def DistributionUpdate(mu_old, Sigma_old, population, no_of_tasks):
    mu = [None] * no_of_tasks
    Sigma = [None] * no_of_tasks
    c_mu = 0.5
    for i in range(no_of_tasks):
        individuals = []
        for ind in population:
            if getattr(ind, 'skill_factor', None) == (i + 1):
                individuals.append(np.asarray(ind.rnvec).ravel())
        if len(individuals) == 0:
            mu[i] = mu_old[i]
            Sigma[i] = Sigma_old[i]
        else:
            arr = np.vstack(individuals)
            mean_arr = np.mean(arr, axis=0)
            cov = np.cov(arr, rowvar=False)
            cov_diag = np.diag(np.diag(cov))
            mu[i] = (1 - c_mu) * mu_old[i] + c_mu * mean_arr
            Sigma[i] = (1 - c_mu) * Sigma_old[i] + c_mu * cov_diag
    return mu, Sigma
