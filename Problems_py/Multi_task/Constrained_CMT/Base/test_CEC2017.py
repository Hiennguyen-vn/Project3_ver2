import numpy as np

# Note: This is a direct-but-concise translation of the MATLAB CEC2017 cases used in the repo.
# The original file has many cases; for full parity use this as reference and extend cases as needed.

def CEC2017(x, I_fno, o, M):
    x = np.atleast_2d(x)
    ps, D = x.shape
    # We'll implement a subset and then follow the same pattern: compute f, g, h and aggregate.
    # For brevity, implement cases 1-4 and a fallback that raises NotImplementedError for others.
    if I_fno == 1:
        y = x - o
        f = np.sum(np.square(np.cumsum(y, axis=1)), axis=1)
        g = np.sum(y ** 2 - 5000.0 * np.cos(0.1 * np.pi * y) - 4000.0, axis=1)
        h = np.zeros(ps)
    elif I_fno == 2:
        y = x - o
        z = (M @ y.T).T if not (isinstance(M, (int, float)) and M == 1) else y
        f = np.sum(np.square(np.cumsum(y, axis=1)), axis=1)
        g = np.sum(z ** 2 - 5000.0 * np.cos(0.1 * np.pi * z) - 4000.0, axis=1)
        h = np.zeros(ps)
    elif I_fno == 3:
        y = x - o
        f = np.sum(np.square(np.cumsum(y, axis=1)), axis=1)
        g = np.sum(y ** 2 - 5000.0 * np.cos(0.1 * np.pi * y) - 4000.0, axis=1)
        h = -np.sum(y * np.sin(0.1 * np.pi * y), axis=1)
    elif I_fno == 4:
        y = x - o
        f = np.sum(y ** 2 - 10 * np.cos(2 * np.pi * y) + 10, axis=1)
        g1 = np.sum(-y * np.sin(2 * y), axis=1)
        g2 = np.sum(y * np.sin(y), axis=1)
        g = np.vstack([g1, g2]).T
        h = np.zeros(ps)
    else:
        raise NotImplementedError(f"CEC2017 case {I_fno} not implemented in this concise port.")

    g[g < 0] = 0
    h = np.abs(h) - 1e-4
    h[h < 0] = 0
    con = np.sum(g, axis=1) + np.sum(h, axis=0)
    # return per-row scalars: f and con
    return f if f.ndim == 1 else f.squeeze(), con if np.isscalar(con) else con
