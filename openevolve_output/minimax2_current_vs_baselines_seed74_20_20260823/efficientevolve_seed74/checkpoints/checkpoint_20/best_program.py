# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Deterministically maximize the minimum squared distance for 16 points of unit diameter."""
    from scipy.optimize import minimize

    n = 16
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)],
                     dtype=int)
    m = len(pairs)

    def separation_constraints(z):
        p = z[:32].reshape(n, 2)
        delta = p[pairs[:, 0]] - p[pairs[:, 1]]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return np.r_[dsq - z[32], 1.0 - dsq]

    def separation_jacobian(z):
        p = z[:32].reshape(n, 2)
        delta = p[pairs[:, 0]] - p[pairs[:, 1]]
        jac = np.zeros((2 * m, 33), dtype=float)
        rows = np.arange(m)
        ixy = 2 * pairs[:, 0]
        jxy = 2 * pairs[:, 1]
        jac[rows, ixy] = 2.0 * delta[:, 0]
        jac[rows, ixy + 1] = 2.0 * delta[:, 1]
        jac[rows, jxy] = -2.0 * delta[:, 0]
        jac[rows, jxy + 1] = -2.0 * delta[:, 1]
        jac[rows, 32] = -1.0
        jac[m:] = -jac[:m]
        jac[m:, 32] = 0.0
        return jac

    def endpoint_equalities(z):
        p = z[:32].reshape(n, 2)
        return np.r_[p[0], p[1] - np.array([1.0, 0.0])]

    def endpoint_jacobian(z):
        jac = np.zeros((4, 33), dtype=float)
        jac[0, 0] = jac[1, 1] = 1.0
        jac[2, 2] = jac[3, 3] = 1.0
        return jac

    def realized_score(p):
        d = p[:, None, :] - p[None, :, :]
        dsq = np.sum(d * d, axis=2)
        np.fill_diagonal(dsq, np.inf)
        return float(np.min(dsq) / np.max(dsq[np.isfinite(dsq)]))

    # A hexagonal lattice is a much stronger seed than a square grid.  The
    # two fixed points are a valid unit-diameter pair, avoiding infeasibility.
    best_points = None
    best_score = -np.inf
    rng = np.random.default_rng(20260823)

    for trial, spacing in enumerate((0.225, 0.235, 0.245, 0.250, 0.242, 0.248)):
        lattice = []
        for row in range(4):
            for col in range(4):
                x = 0.5 + (col - 1.5 + 0.5 * (row & 1)) * spacing
                y = (row - 1.5) * np.sqrt(3.0) * spacing / 2.0
                lattice.append((x, y))

        # Remove different interior lattice sites in each restart, then use
        # the two remaining slots for the fixed diameter endpoints.
        remove = ((5 + trial) % 16, (10 + 3 * trial) % 16)
        interior = np.array([v for k, v in enumerate(lattice) if k not in remove],
                            dtype=float)
        interior += rng.normal(0.0, 0.004 + 0.001 * trial, interior.shape)
        points0 = np.vstack(([[0.0, 0.0], [1.0, 0.0]], interior))
        delta0 = points0[:, None, :] - points0[None, :, :]
        dsq0 = np.sum(delta0 * delta0, axis=2)
        np.fill_diagonal(dsq0, np.inf)
        z0 = np.r_[points0.ravel(), 0.995 * np.min(dsq0)]

        result = minimize(
            lambda z: -z[32],
            z0,
            jac=lambda z: np.r_[np.zeros(32), -1.0],
            method="SLSQP",
            bounds=[(-0.08, 1.08), (-0.62, 0.62)] * n + [(0.0, 1.0)],
            constraints=[
                {"type": "ineq", "fun": separation_constraints,
                 "jac": separation_jacobian},
                {"type": "eq", "fun": endpoint_equalities,
                 "jac": endpoint_jacobian},
            ],
            options={"maxiter": 1800, "ftol": 1e-12, "disp": False},
        )

        if np.all(np.isfinite(result.x)):
            candidate = result.x[:32].reshape(n, 2)
            score = realized_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate

    if best_points is None:
        best_points = np.array([[0.0, 0.0], [1.0, 0.0]] +
                               [[0.5 + 0.2 * (k % 4 - 1.5),
                                 0.18 * (k // 4 - 1.5)]
                                for k in range(14)], dtype=float)

    delta = best_points[:, None, :] - best_points[None, :, :]
    diameter = np.sqrt(np.max(np.sum(delta * delta, axis=2)))
    return np.asarray(best_points / diameter, dtype=float)


# EVOLVE-BLOCK-END
