import numpy as np


def AT_Transfer(source_solution, mu_s, Sigma_s, mu_t, Sigma_t):
    """Affine transfer from source to target domain.

    Parameters expected as numpy arrays. Sigma_* are (diagonal) covariance matrices or 1D arrays.
    """
    # Ensure arrays
    source_solution = np.asarray(source_solution)
    mu_s = np.asarray(mu_s)
    mu_t = np.asarray(mu_t)
    Sigma_s = np.asarray(Sigma_s)
    Sigma_t = np.asarray(Sigma_t)

    # Ensure Sigma are full matrices
    if Sigma_s.ndim == 1:
        Sigma_s_mat = np.diag(Sigma_s)
    else:
        Sigma_s_mat = Sigma_s
    if Sigma_t.ndim == 1:
        Sigma_t_mat = np.diag(Sigma_t)
    else:
        Sigma_t_mat = Sigma_t

    # Compute inverse cholesky factors with safe fallback
    try:
        Lsi_l = np.linalg.cholesky(np.linalg.inv(Sigma_s_mat))
    except Exception:
        Lsi_l = np.linalg.cholesky(np.linalg.pinv(Sigma_s_mat))
    try:
        Lci_l = np.linalg.cholesky(np.linalg.inv(Sigma_t_mat))
    except Exception:
        Lci_l = np.linalg.cholesky(np.linalg.pinv(Sigma_t_mat))

    Am_l = np.linalg.inv(Lci_l.T).dot(Lsi_l)
    bm_l = mu_t.reshape(-1, 1) - Am_l.dot(mu_s.reshape(-1, 1))
    solution_transfer = (Am_l.dot(source_solution.reshape(-1, 1)) + bm_l).ravel()
    return solution_transfer
