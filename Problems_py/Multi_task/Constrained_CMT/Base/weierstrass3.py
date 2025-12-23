import numpy as np

def _apply_M(var, M, opt, dim):
    v = var[:dim] - opt[:dim]
    if hasattr(M, 'shape') and M is not None and not (isinstance(M, (int, float)) and M == 1):
        x = M.dot(v)
    else:
        x = v
    return x


def weierstrass3(var, M, opt, opt_con):
    var = np.asarray(var).flatten()
    dim = var.size
    x = _apply_M(var, M, opt, dim)
    a = 0.5
    b = 3
    kmax = 20
    obj = 0.0
    for i in range(dim):
        for k in range(kmax + 1):
            obj += a ** k * np.cos(2 * np.pi * (b ** k) * (x[i] + 0.5))
    for k in range(kmax + 1):
        obj -= dim * a ** k * np.cos(2 * np.pi * (b ** k) * 0.5)

    x2 = 200 * (var - opt_con[:dim])
    g1 = -np.sum(np.abs(x2)) + 12 * dim
    g2 = np.sum(x2 ** 2) - 500 * dim
    g = np.array([g1, g2])
    h = 0.0

    g[g < 0] = 0
    h = np.abs(h) - 1e-4
    if h < 0:
        h = 0.0
    con = np.sum(g) + h
    return obj, con
