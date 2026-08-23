# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Find a diameter-one 16-point packing by deterministic multistart SLSQP.

    A triangular lattice is perturbed at several scales, supplemented by
    independent cloud starts, and the best feasible local packing is polished.
    """
    from scipy.optimize import minimize

    n = 16
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)

    def pair_sq(x):
        p = x[: 2 * n].reshape(n, 2)
        delta = p[ii] - p[jj]
        return np.einsum("ij,ij->i", delta, delta)

    def constraints(x):
        dsq = pair_sq(x)
        t = x[-1]
        # Lower constraints enforce pair distance >= t.
        # Upper constraints enforce diameter <= 1.
        return np.concatenate((dsq - t * t, 1.0 - dsq))

    def constraints_jac(x):
        p = x[: 2 * n].reshape(n, 2)
        t = x[-1]
        delta = p[ii] - p[jj]

        jac = np.zeros((2 * m, 2 * n + 1))
        rows = np.arange(m)

        jac[rows, 2 * ii] = 2.0 * delta[:, 0]
        jac[rows, 2 * ii + 1] = 2.0 * delta[:, 1]
        jac[rows, 2 * jj] = -2.0 * delta[:, 0]
        jac[rows, 2 * jj + 1] = -2.0 * delta[:, 1]
        jac[rows, -1] = -2.0 * t

        upper = rows + m
        jac[upper, 2 * ii] = -2.0 * delta[:, 0]
        jac[upper, 2 * ii + 1] = -2.0 * delta[:, 1]
        jac[upper, 2 * jj] = 2.0 * delta[:, 0]
        jac[upper, 2 * jj + 1] = 2.0 * delta[:, 1]
        return jac

    def centroid_constraint(x):
        return np.sum(x[: 2 * n].reshape(n, 2), axis=0)

    def centroid_jac(x):
        jac = np.zeros((2, 2 * n + 1))
        jac[0, 0:2 * n:2] = 1.0
        jac[1, 1:2 * n:2] = 1.0
        return jac

    # A compact triangular-lattice patch is a substantially better starting
    # point than independent random Gaussian coordinates.
    base = np.array(
        [[col + 0.5 * (row & 1), np.sqrt(3.0) * 0.5 * row]
         for row in range(4) for col in range(4)],
        dtype=float,
    )
    base -= base.mean(axis=0)

    rng = np.random.default_rng(20260823)
    best_points = None
    best_ratio = -np.inf

    nonlinear = {
        "type": "ineq",
        "fun": constraints,
        "jac": constraints_jac,
    }
    centered = {
        "type": "eq",
        "fun": centroid_constraint,
        "jac": centroid_jac,
    }

    # Small perturbations retain good lattice contacts, while large
    # perturbations and cloud starts permit transitions to different basins.
    for restart in range(48):
        if restart == 0:
            p = base.copy()
        elif restart < 40:
            scale = 0.035 + 0.0125 * restart
            p = base + rng.normal(scale=scale, size=(n, 2))
        else:
            p = rng.uniform(-1.0, 1.0, size=(n, 2))
        p -= p.mean(axis=0)

        d = p[:, None, :] - p[None, :, :]
        diameter = np.sqrt(np.max(np.sum(d * d, axis=2)))
        p *= 0.985 / diameter

        dsq = pair_sq(np.concatenate((p.ravel(), [0.0])))
        x0 = np.concatenate((p.ravel(), [0.90 * np.sqrt(np.min(dsq))]))

        result = minimize(
            fun=lambda x: -x[-1],
            x0=x0,
            jac=lambda x: np.r_[np.zeros(2 * n), -1.0],
            method="SLSQP",
            bounds=[(-1.0, 1.0)] * (2 * n) + [(0.0, 1.0)],
            constraints=[nonlinear, centered],
            options={"maxiter": 2400, "ftol": 2e-12, "disp": False},
        )

        p = result.x[: 2 * n].reshape(n, 2)
        d = p[:, None, :] - p[None, :, :]
        distances = np.sqrt(np.sum(d * d, axis=2))
        distances[np.diag_indices(n)] = np.inf
        ratio = np.min(distances) / np.max(distances[np.isfinite(distances)])

        if ratio > best_ratio:
            best_ratio = ratio
            best_points = p

    # Re-solve from the best basin with tighter termination.  Normalizing
    # before this call makes every upper-distance constraint feasible exactly.
    d = best_points[:, None, :] - best_points[None, :, :]
    diameter = np.sqrt(np.max(np.sum(d * d, axis=2)))
    best_points = best_points / diameter
    dsq = pair_sq(np.concatenate((best_points.ravel(), [0.0])))
    x0 = np.concatenate((best_points.ravel(), [0.999 * np.sqrt(np.min(dsq))]))
    polished = minimize(
        fun=lambda x: -x[-1],
        x0=x0,
        jac=lambda x: np.r_[np.zeros(2 * n), -1.0],
        method="SLSQP",
        bounds=[(-1.0, 1.0)] * (2 * n) + [(0.0, 1.0)],
        constraints=[nonlinear, centered],
        options={"maxiter": 8000, "ftol": 1e-13, "disp": False},
    )

    candidate = polished.x[: 2 * n].reshape(n, 2)
    d = candidate[:, None, :] - candidate[None, :, :]
    candidate_diameter = np.sqrt(np.max(np.sum(d * d, axis=2)))
    candidate /= candidate_diameter
    d = candidate[:, None, :] - candidate[None, :, :]
    distances = np.sqrt(np.sum(d * d, axis=2))
    distances[np.diag_indices(n)] = np.inf
    candidate_ratio = np.min(distances)

    d = best_points[:, None, :] - best_points[None, :, :]
    distances = np.sqrt(np.sum(d * d, axis=2))
    distances[np.diag_indices(n)] = np.inf
    if candidate_ratio > np.min(distances):
        best_points = candidate

    return best_points


# EVOLVE-BLOCK-END
