import matplotlib.pyplot as plt
import numpy as np

# Data
problems = ['CMT1', 'CMT2', 'CMT3', 'CMT4', 'CMT5', 'CMT6', 'CMT7', 'CMT8', 'CMT9']
n_problems = len(problems)

# Paper Results (RL-CMTEA from Table 3)
paper_t1 = [4.81e-17, 2.19e-09, 2.28e-04, 8.79e+01, 4.29e-12, 1.79e-08, 1.13e+04, 1.61e+01, 1.94e+01]
paper_t2 = [7.98e-14, 5.92e-17, 1.30e-03, 8.15e+02, 9.74e+01, 6.60e-05, 1.29e+02, 9.19e+01, 3.32e+04]

# Our DaS Results (Best values)
# CMT1: 0.00 -> use 1e-17 for log scale
das_t1 = [1e-17, 4.44e-16, 2.89e-14, 1e-17, 4.44e-16, 4.44e-16, 1.75e+00, 6.00e+00, 4.69e+03]
das_t2 = [1e-17, 1e-17, 6.36e-04, 3.76e+02, 4.00e+01, 1e-18, 6.08e+01, 4.23e+01, 1.66e+04]

# Plotting
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Task 1
x = np.arange(n_problems)

axes[0].plot(x, paper_t1, marker='o', linestyle='-', label='Paper (RL-CMTEA)', color='#1f77b4', linewidth=2)
axes[0].plot(x, das_t1, marker='s', linestyle='--', label='Ours (DaS KT)', color='#ff7f0e', linewidth=2)

axes[0].set_ylabel('Objective Value (Log Scale)')
axes[0].set_title('Task 1 Performance Comparison (Lower is Better)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(problems)
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True, which="both", ls="-", alpha=0.2)

# Task 2
axes[1].plot(x, paper_t2, marker='o', linestyle='-', label='Paper (RL-CMTEA)', color='#1f77b4', linewidth=2)
axes[1].plot(x, das_t2, marker='s', linestyle='--', label='Ours (DaS KT)', color='#ff7f0e', linewidth=2)

axes[1].set_ylabel('Objective Value (Log Scale)')
axes[1].set_title('Task 2 Performance Comparison (Lower is Better)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(problems)
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.savefig('comparison_cmt1_9_line.png', dpi=300)
print("Saved comparison line chart to comparison_cmt1_9_line.png")
