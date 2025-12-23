class Individual:
    def __init__(self):
        self.rnvec = None  # unified gene in [0,1]
        self.factorial_costs = None
        self.constraint_violation = None
        self.factorial_ranks = None
        self.scalar_fitness = None
        self.skill_factor = None
