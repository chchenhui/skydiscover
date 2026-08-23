# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Minimize diameter of a unit-separated 16-point set by multiscale SLSQP restarts.

    A triangular-lattice seed is perturbed at several deterministic scales,
    normalized back to unit minimum separation, and locally optimized with all
    pairwise separation and diameter constraints enforced explicitly.
    """
    from scipy.optimize import minimize

    axial = np.array(
        [
            [0, -2], [1, -2],
            [-1, -1], [0, -1], [1, -1], [2, -1],
            [-2, 0], [-1, 0], [0, 0], [1, 0],
            [-2, 1], [-1, 1], [0, 1], [1, 1],
            [-2, 2], [-1, 2],
        ],
        dtype=np.float64,
    )
    base = np.column_stack(
        (axial[:, 0] + 0.5 * axial[:, 1],
         0.5 * np.sqrt(3.0) * axial[:, 1])
    )
    base -= base.mean(axis=0)

    ii, jj = np.triu_indices(16, 1)
    m = len(ii)

    def distances(z):
        p = z[:-1].reshape(16, 2)
        d = p[ii] - p[jj]
        return np.einsum("ij,ij->i", d, d), d

    def constraints(z):
        d2, _ = distances(z)
        return np.concatenate((d2 - 1.0, z[-1] - d2))

    def constraint_jacobian(z):
        _, d = distances(z)
        jac = np.zeros((2 * m, 33), dtype=np.float64)
        row = np.arange(m)
        jac[row, 2 * ii] = 2.0 * d[:, 0]
        jac[row, 2 * ii + 1] = 2.0 * d[:, 1]
        jac[row, 2 * jj] = -2.0 * d[:, 0]
        jac[row, 2 * jj + 1] = -2.0 * d[:, 1]
        jac[m:] = -jac[:m]
        jac[m + row, -1] = 1.0
        return jac

    rng = np.random.default_rng(271828)
    best = base.copy()
    best_d2 = np.sum((best[:, None] - best[None, :]) ** 2, axis=2)
    best_ratio = best_d2[np.triu_indices(16, 1)].min() / best_d2.max()

    # The lattice itself is a useful incumbent, but its exact contact graph is
    # a local basin.  Use several perturbation radii to visit neighboring and
    # more strongly rearranged contact graphs deterministically.
    perturbation_scales = (0.025, 0.050, 0.085, 0.135, 0.210, 0.300)
    for restart in range(192):
        if restart == 0:
            p = base.copy()
        else:
            scale = perturbation_scales[(restart - 1) % len(perturbation_scales)]
            p = base + scale * rng.standard_normal((16, 2))
        p -= p.mean(axis=0)
        d2 = np.sum((p[:, None] - p[None, :]) ** 2, axis=2)
        p /= np.sqrt(d2[np.triu_indices(16, 1)].min())
        d2 = np.sum((p[:, None] - p[None, :]) ** 2, axis=2)
        z0 = np.concatenate((p.ravel(), [d2.max()]))

        result = minimize(
            lambda z: z[-1],
            z0,
            jac=lambda z: np.r_[np.zeros(32), 1.0],
            constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
            method="SLSQP",
            options={"maxiter": 1200, "ftol": 3e-13, "disp": False},
        )

        p = result.x[:-1].reshape(16, 2)
        p -= p.mean(axis=0)
        d2 = np.sum((p[:, None] - p[None, :]) ** 2, axis=2)
        pair_d2 = d2[np.triu_indices(16, 1)]
        p /= np.sqrt(pair_d2.min())
        d2 = np.sum((p[:, None] - p[None, :]) ** 2, axis=2)
        ratio = d2[np.triu_indices(16, 1)].min() / d2.max()

        if ratio > best_ratio:
            best, best_ratio = p, ratio

    return best


# EVOLVE-BLOCK-END
