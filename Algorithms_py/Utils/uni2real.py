def uni2real(X, Tasks):
    """Map unified [0,1] rnvec to real bound for a list of bestX.
    X: list where X[t] is rnvec for task t
    Tasks: list-like of task objects with Lb, Ub, dims
    Returns mapped list X (in-place mapping)
    """
    out = [None] * len(Tasks)
    for t in range(len(Tasks)):
        x = X[t]
        if x is None:
            out[t] = None
            continue
        out[t] = Tasks[t].Lb + x[:Tasks[t].dims] * (Tasks[t].Ub - Tasks[t].Lb)
    return out
