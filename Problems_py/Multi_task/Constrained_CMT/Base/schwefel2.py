import numpy as np

def _apply_M(var, M, opt, dim):
    v = var[:dim] - opt[:dim]
    if hasattr(M, 'shape') and M is not None and not (isinstance(M, (int, float)) and M == 1):
        x = M.dot(v)
    else:
        x = v
    return x


def schwefel2(var, M, opt, opt_con):
    var = np.asarray(var).flatten()
    dim = var.size
    x = _apply_M(var, M, opt, dim)
    sumx = np.sum(x * np.sin(np.sqrt(np.abs(x))))
    obj = 418.9829 * dim - sumx

    x2 = 0.2 * (var - opt_con[:dim])
    g = np.sum(x2 ** 2) - 100 * dim
    h = 0.0

    g = max(g, 0.0)
    h = abs(h) - 1e-4
    h = max(h, 0.0)
    con = g + h
    return obj, con
