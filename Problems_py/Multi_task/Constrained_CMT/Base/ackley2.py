import numpy as np

def _apply_M(var, M, opt, dim):
    v = var[:dim] - opt[:dim]
    if hasattr(M, 'shape') and M is not None and not (isinstance(M, (int, float)) and M == 1):
        x = M.dot(v)
    else:
        x = v
    return x


def ackley2(var, M, opt, opt_con):
    var = np.asarray(var).flatten()
    dim = var.size
    x = _apply_M(var, M, opt, dim)

    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    avgsum1 = sum1 / dim
    avgsum2 = sum2 / dim
    obj = -20 * np.exp(-0.2 * np.sqrt(avgsum1)) - np.exp(avgsum2) + 20 + np.e

    # constraint
    x2 = 2 * (var - opt_con[:dim])
    g = np.sum(x2 ** 2) - 100 * dim
    h = 0.0

    g = max(g, 0.0)
    h = abs(h) - 1e-4
    h = max(h, 0.0)
    con = g + h
    return obj, con
