import numpy as np

def _apply_M(var, M, opt, dim):
    v = var[:dim] - opt[:dim]
    if hasattr(M, 'shape') and M is not None and not (isinstance(M, (int, float)) and M == 1):
        x = M.dot(v)
    else:
        x = v
    return x


def rosenbrock1(var, M, opt, opt_con):
    var = np.asarray(var).flatten()
    dim = var.size
    x = _apply_M(var, M, opt, dim)
    sumx = 0.0
    if dim == 1:
        sumx = 100 * (x[0] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2
    else:
        for ii in range(dim - 1):
            xi = x[ii]
            xnext = x[ii + 1]
            sumx += 100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2
    obj = sumx

    x2 = 2 * (var - opt_con[:dim])
    g = np.sum(x2 ** 2 - 5000.0 * np.cos(0.1 * np.pi * x2) - 4000.0)
    h = 0.0

    g = max(g, 0.0)
    h = abs(h) - 1e-4
    h = max(h, 0.0)
    con = g + h
    return obj, con
