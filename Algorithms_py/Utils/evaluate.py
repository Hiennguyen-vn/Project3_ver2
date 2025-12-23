import numpy as np

def evaluate(population, Task, task_idx):
    """Evaluate a population on a single Task.
    population: list of Individual
    Task: object with dims, Lb, Ub, fnc (callable)
    task_idx: index (unused in Python version but kept for parity)
    Returns (population, calls)
    """
    calls = 0
    for i in range(len(population)):
        indiv = population[i]
        rn = indiv.rnvec[:Task.dims]
        x = Task.Lb + rn * (Task.Ub - Task.Lb)
        obj, con = Task.fnc(x)
        if indiv.factorial_costs is None:
            indiv.factorial_costs = [float('inf')] * task_idx
        # ensure lists big enough
        if not isinstance(indiv.factorial_costs, list):
            indiv.factorial_costs = list(indiv.factorial_costs)
        # expand if necessary
        if len(indiv.factorial_costs) < task_idx:
            indiv.factorial_costs += [float('inf')] * (task_idx - len(indiv.factorial_costs))
        if indiv.constraint_violation is None:
            indiv.constraint_violation = [float('inf')] * task_idx
        if not isinstance(indiv.constraint_violation, list):
            indiv.constraint_violation = list(indiv.constraint_violation)
        if len(indiv.constraint_violation) < task_idx:
            indiv.constraint_violation += [float('inf')] * (task_idx - len(indiv.constraint_violation))

        # assign
        # convert to python float
        val = float(obj)
        con_val = float(con)
        # ensure length at least task_idx
        if len(indiv.factorial_costs) >= task_idx:
            if task_idx - 1 < len(indiv.factorial_costs):
                indiv.factorial_costs[task_idx - 1] = val
            else:
                indiv.factorial_costs.append(val)
        else:
            indiv.factorial_costs.append(val)
        if len(indiv.constraint_violation) >= task_idx:
            if task_idx - 1 < len(indiv.constraint_violation):
                indiv.constraint_violation[task_idx - 1] = con_val
            else:
                indiv.constraint_violation.append(con_val)
        else:
            indiv.constraint_violation.append(con_val)
        calls += 1
    return population, calls
