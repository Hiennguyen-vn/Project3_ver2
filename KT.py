# KT.py
from dataclasses import dataclass
import numpy as np
import copy
import os
from typing import List, Optional

# ---------------------------------------------------------
# Cấu trúc dữ liệu
# ---------------------------------------------------------
@dataclass
class Individual:
    rnvec: np.ndarray          # [0,1]^D
    factorial_costs: float = None
    constraint_violation: float = None

@dataclass
class AlgoParams:
    DE_F: float = 0.5
    DE_CR: float = 0.9

# ---------------------------------------------------------
# KMeans: sklearn nếu có, fallback NumPy
# ---------------------------------------------------------
_USE_SKLEARN = os.environ.get("RL_CMTEA_USE_SKLEARN", "").lower() in {"1", "true", "yes"}

def _kmeans_numpy_fallback(X, n_clusters, n_init=5, max_iter=100, rng=None):
    rng = np.random.default_rng(rng)
    best_labels, best_inertia = None, None
    for _ in range(n_init):
        centroids = X[rng.choice(len(X), size=n_clusters, replace=False)].copy()
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - centroids[None, :, :])**2).sum(axis=2)
            labels = d2.argmin(axis=1)
            new_centroids = np.vstack([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                for k in range(n_clusters)
            ])
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        inertia = ((X - centroids[labels])**2).sum()
        if best_inertia is None or inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels

def kmeans_labels(X, n_clusters, rng=None):
    if _USE_SKLEARN:
        try:
            from sklearn.cluster import KMeans
            seed = int(np.random.default_rng(rng).integers(0, 2**31 - 1))
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
            return km.fit_predict(X)
        except Exception:
            pass
    return _kmeans_numpy_fallback(X, n_clusters, rng=rng)


def _balance_clusters_kl(labels, n_clusters, rng, X, max_retries=3, kl_threshold=0.5):
    """Retry kmeans assignments if clusters are too imbalanced (KL vs. uniform).

    This is a lightweight guard: compute KL(P || U) with P = cluster freq / N,
    U = uniform. If KL > threshold, re-run kmeans a few times to seek a more
    balanced split (helps KT avoid collapsing clusters).
    """
    counts = np.bincount(labels, minlength=n_clusters).astype(float)
    N = counts.sum()
    if N == 0:
        return labels
    p = counts / N
    u = np.full(n_clusters, 1.0 / n_clusters)
    kl = float(np.sum(np.where(p > 0, p * np.log(p / u), 0.0)))
    if kl <= kl_threshold:
        return labels

    best_labels, best_kl = labels, kl
    for _ in range(max_retries):
        new_labels = kmeans_labels(X, n_clusters, rng=rng)
        counts = np.bincount(new_labels, minlength=n_clusters).astype(float)
        p = counts / counts.sum()
        kl_new = float(np.sum(np.where(p > 0, p * np.log(p / u), 0.0)))
        if kl_new < best_kl:
            best_labels, best_kl = new_labels, kl_new
        if best_kl <= kl_threshold:
            break
    return best_labels

# ---------------------------------------------------------
# DE binomial crossover
# ---------------------------------------------------------
def de_crossover(v: np.ndarray, u: np.ndarray, CR: float, rng=None) -> np.ndarray:
    rng = np.random.default_rng(rng)
    D = v.shape[-1]
    jrand = rng.integers(0, D)
    mask = rng.random(D) < CR
    mask[jrand] = True
    trial = np.where(mask, v, u)
    return trial

# ---------------------------------------------------------
# correDecode: MATLAB 1-based → Python 0-based, padding tới divD
# ---------------------------------------------------------
def corre_decode(population: List[List[Individual]], corre_row: np.ndarray, divD: int) -> np.ndarray:
    # corre_row = [t, i, start, end] (1-based trong MATLAB)
    t, i, start, end = map(int, corre_row)
    t_idx, i_idx = t - 1, i - 1
    start_idx, end_idx = start - 1, end  # slice exclusive
    seg = population[t_idx][i_idx].rnvec[start_idx:end_idx]
    if seg.size == divD:
        return seg.copy()
    out = np.zeros(divD, dtype=float)
    out[:seg.size] = seg
    return out

# ---------------------------------------------------------
# Hàm KT chính (port 1:1 từ MATLAB)
# ---------------------------------------------------------
def KT(Algo: AlgoParams,
    Tasks_dims: List[int],
    population: List[List[Individual]],
    divK: int,
    divD: int,
    rng: Optional[np.random.Generator] = None,
    kl_balance: bool = False,
    kl_threshold: float = 0.5,
    return_meta: bool = False) -> List[List[Individual]]:
    """
    Algo: AlgoParams(DE_F, DE_CR)
    Tasks_dims: list[int] chứa dim của từng task (tương ứng MATLAB: Tasks.dims)
    population: list[list[Individual]]
    divK: số cụm kmeans
    divD: độ dài mỗi block
    """
    rng = np.random.default_rng(rng)

    # ------------------------------
    # 1) Xây corre theo max(Tasks.dims)
    #    Mỗi hàng: [task, ind, start, end] (MATLAB 1-based)
    # ------------------------------
    max_dim = int(max(Tasks_dims))
    corre_rows = []
    for t in range(1, len(Tasks_dims) + 1):              # 1-based
        n_ind = len(population[t - 1])
        for i in range(1, n_ind + 1):                    # 1-based
            n_blocks = int(np.ceil(max_dim / divD))
            for j in range(1, n_blocks + 1):             # 1-based
                start = 1 + (j - 1) * divD
                end = min(j * divD, max_dim)
                corre_rows.append([t, i, start, end])
    corre = np.array(corre_rows, dtype=int)

    # ------------------------------
    # 2) dimVal: giải mã từng block (padding tới divD)
    # ------------------------------
    dimVal = np.vstack([corre_decode(population, row, divD) for row in corre])

    # ------------------------------
    # 3) KMeans để chia cụm
    # ------------------------------
    idx = kmeans_labels(dimVal, divK, rng=rng)
    if kl_balance and divK > 1 and len(dimVal) > 0:
        idx = _balance_clusters_kl(idx, divK, rng, dimVal, max_retries=3, kl_threshold=kl_threshold)
    subpop = [[] for _ in range(divK)]
    for r, k in enumerate(idx):
        subpop[int(k)].append(r)  # lưu chỉ số hàng trong corre/dimVal

    # ------------------------------
    # 4) DE/rand/1 binomial trong từng cụm
    # ------------------------------
    offspring_temp = []
    off_corre = []
    off_src = []

    for k in range(divK):
        members = subpop[k]
        m = len(members)
        for i_local, row_idx in enumerate(members):
            if m < 4:
                continue
            # lấy 4 chỉ số khác nhau trong [0, m); loại bản thân nếu trúng
            A_local = list(rng.choice(m, size=4, replace=False))
            A_global = [members[a] for a in A_local if members[a] != row_idx]
            if len(A_global) < 3:
                # fallback: lấy 3 khác row_idx
                cand = [g for g in members if g != row_idx]
                if len(cand) < 3:
                    continue
                A_global = list(rng.choice(cand, size=3, replace=False))
            r1, r2, r3 = A_global[:3]
            dp1 = corre_decode(population, corre[r1], divD)
            dp2 = corre_decode(population, corre[r2], divD)
            dp3 = corre_decode(population, corre[r3], divD)

            # mutant & clip
            v = dp1 + Algo.DE_F * (dp2 - dp3)
            np.clip(v, 0.0, 1.0, out=v)

            # target u là block hiện tại
            u = corre_decode(population, corre[row_idx], divD)
            trial = de_crossover(v, u, Algo.DE_CR, rng=rng)
            np.clip(trial, 0.0, 1.0, out=trial)

            offspring_temp.append(trial)
            off_corre.append(corre[row_idx])
            # record primary source task (use r1's task as representative)
            try:
                src_task = int(corre[r1][0]) - 1
            except Exception:
                src_task = -1
            off_src.append(src_task)

    offspring_temp = np.array(offspring_temp) if offspring_temp else np.empty((0, divD))
    off_corre = np.array(off_corre, dtype=int) if off_corre else np.empty((0, 4), dtype=int)
    off_src = np.array(off_src, dtype=int) if off_src else np.empty((0,), dtype=int)

    # ------------------------------
    # 5) Ghi về offspring (deepcopy)
    #    LƯU Ý: end có thể vượt dim thật của task → cắt theo lát cắt thực tế
    # ------------------------------
    offspring = copy.deepcopy(population)
    for i in range(off_corre.shape[0]):
        t, ind, start, end = off_corre[i]
        t_idx, ind_idx = t - 1, ind - 1
        start_idx, end_idx = start - 1, end
        view = offspring[t_idx][ind_idx].rnvec[start_idx:end_idx]
        actual_len = view.size
        view[:] = offspring_temp[i][:actual_len]
        np.clip(offspring[t_idx][ind_idx].rnvec, 0.0, 1.0, out=offspring[t_idx][ind_idx].rnvec)

    if return_meta:
        # off_src is 0-based task indices corresponding to each offspring_temp row
        return offspring, off_corre, off_src, offspring_temp
    return offspring