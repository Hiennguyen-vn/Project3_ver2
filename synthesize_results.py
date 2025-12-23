
import json
import numpy as np

# Paper results from Table 3
PAPER_RESULTS = {
    'CMT1': {'T1': 4.81e-17, 'T2': 7.98e-14},
    'CMT2': {'T1': 2.19e-09, 'T2': 5.92e-17},
    'CMT3': {'T1': 2.28e-04, 'T2': 1.30e-03},
    'CMT4': {'T1': 8.79e+01, 'T2': 8.15e+02},
    'CMT5': {'T1': 4.29e-12, 'T2': 9.740e+01},
    'CMT6': {'T1': 1.79e-08, 'T2': 6.60e-05},
    'CMT7': {'T1': 1.13e+04, 'T2': 1.29e+02},
    'CMT8': {'T1': 1.61e+01, 'T2': 9.19e+01},
    'CMT9': {'T1': 1.94e+01, 'T2': 3.32e+04},
}

# Results for runs 1-10 (from previous turn logs)
RUNS_1_10_STATS = {
    'CMT1': {'t1_mean': 1.1102e-17, 't1_best': 0.0,         't2_mean': 5.9698e-01, 't2_best': 0.0},
    'CMT2': {'t1_mean': 1.3058e-10, 't1_best': 4.4409e-16, 't2_mean': 0.0,        't2_best': 0.0},
    'CMT3': {'t1_mean': 4.4316e-09, 't1_best': 2.8866e-14, 't2_mean': 6.3638e-04, 't2_best': 6.3638e-04},
    'CMT4': {'t1_mean': 8.8201e+00, 't1_best': 0.0,         't2_mean': 3.7901e+02, 't2_best': 376.34},
    'CMT5': {'t1_mean': 1.9454e+00, 't1_best': 4.4409e-16, 't2_mean': 4.7075e+01, 't2_best': 45.03},
    'CMT6': {'t1_mean': 3.9879e-14, 't1_best': 4.4409e-16, 't2_mean': -3.9777e-18, 't2_best': -3.9777e-18},
    'CMT7': {'t1_mean': 6.4746e+02, 't1_best': 2.2222e+00, 't2_mean': 6.2284e+01, 't2_best': 60.692},
    'CMT8': {'t1_mean': 6.0007e+00, 't1_best': 6.0003e+00, 't2_mean': 4.3113e+01, 't2_best': 42.310},
    'CMT9': {'t1_mean': 7.7300e+03, 't1_best': 6.3370e+03, 't2_mean': 1.6594e+04, 't2_best': 16594.0},
}

with open('/Users/nguyenhien/Prj3-ver2/cmt_results_runs_11_30.json', 'r') as f:
    runs_11_30 = json.load(f)

print(f"{'Problem':<8} {'Task':<5} {'Paper':<12} {'DaS (30-run Mean)':<18} {'DaS (30-run Best)':<18} {'Improvement':<12} {'Winner':<8}")
print("-" * 95)

for name in PAPER_RESULTS:
    r10 = RUNS_1_10_STATS[name]
    r20 = runs_11_30[name]
    
    # Calculate 30-run mean
    t1_mean_30 = (r10['t1_mean'] * 10 + np.mean(r20['t1']) * 20) / 30
    t2_mean_30 = (r10['t2_mean'] * 10 + np.mean(r20['t2']) * 20) / 30
    
    # Calculate 30-run best
    t1_best_30 = min(r10['t1_best'], np.min(r20['t1']))
    t2_best_30 = min(r10['t2_best'], np.min(r20['t2']))
    
    paper = PAPER_RESULTS[name]
    
    # T1
    imp1 = ((paper['T1'] - t1_mean_30) / paper['T1']) * 100 if paper['T1'] != 0 else 0
    win1 = "DaS" if t1_mean_30 < paper['T1'] else "Paper"
    print(f"{name:<8} {'T1':<5} {paper['T1']:<12.2e} {t1_mean_30:<18.4e} {t1_best_30:<18.4e} {imp1:>+10.1f}% {win1:<8}")
    
    # T2
    imp2 = ((paper['T2'] - t2_mean_30) / paper['T2']) * 100 if paper['T2'] != 0 else 0
    win2 = "DaS" if t2_mean_30 < paper['T2'] else "Paper"
    print(f"{'':<8} {'T2':<5} {paper['T2']:<12.2e} {t2_mean_30:<18.4e} {t2_best_30:<18.4e} {imp2:>+10.1f}% {win2:<8}")

# Summary of wins
wins_t1 = 0
wins_t2 = 0
for name in PAPER_RESULTS:
    r10 = RUNS_1_10_STATS[name]
    r20 = runs_11_30[name]
    t1_mean_30 = (r10['t1_mean'] * 10 + np.mean(r20['t1']) * 20) / 30
    t2_mean_30 = (r10['t2_mean'] * 10 + np.mean(r20['t2']) * 20) / 30
    if t1_mean_30 < PAPER_RESULTS[name]['T1']: wins_t1 += 1
    if t2_mean_30 < PAPER_RESULTS[name]['T2']: wins_t2 += 1

print("\n" + "="*95)
print(f"TOTAL WINS: T1={wins_t1}/9, T2={wins_t2}/9 (Total: {wins_t1+wins_t2}/18 tasks)")
print("="*95)
