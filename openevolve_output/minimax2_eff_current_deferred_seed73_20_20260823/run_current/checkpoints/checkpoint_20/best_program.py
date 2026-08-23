# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Deterministically optimize 16 points with fixed unit diameter.

    The two opposite-corner anchor points remove translation, rotation, and
    scale degeneracies. SLSQP maximizes a common lower bound on all pairwise
    distances while constraining every distance to be at most one.
    """
    from scipy.optimize import minimize

    n = 16

    # Deterministic 4x4 starting configuration, normalized to unit diameter.
    yy, xx = np.divmod(np.arange(n), 4)
    initial = np.column_stack((xx.astype(float), yy.astype(float))) / np.sqrt(18.0)

    # Fix two opposite points at distance one. The remaining 14 points are
    # optimization variables, followed by the minimum-distance lower bound t.
    variable_indices = np.arange(1, n - 1)
    def unpack(z):
        """Insert free coordinates between two fixed unit-distance anchors."""
        points = np.empty((n, 2), dtype=float)
        points[0] = (0.0, 0.0)
        points[-1] = (1.0, 0.0)
        points[variable_indices] = z[:-1].reshape(-1, 2)
        return points

    pairs = np.array(
        [(i, j) for i in range(n) for j in range(i + 1, n)],
        dtype=int,
    )

    def squared_distances(z):
        """Return all pairwise squared distances for a candidate vector."""
        points = unpack(z)
        delta = points[pairs[:, 0]] - points[pairs[:, 1]]
        return np.einsum("ij,ij->i", delta, delta)

    def objective(z):
        """Maximize the common pairwise-distance lower bound."""
        return -z[-1]

    def lower_constraints(z):
        """Require every pairwise distance to exceed the lower bound."""
        return squared_distances(z) - z[-1] * z[-1]

    def upper_constraints(z):
        """Keep every distance below the anchored unit diameter."""
        return 1.0 - squared_distances(z)

    # Use several deterministic basins.  The staggered and jittered starts
    # substantially reduce the chance of retaining a poor grid local optimum.
    seeds = []
    for row in range(4):
        for col in range(4):
            k = 4 * row + col
            if k not in (0, 15):
                x = col / 3.0
                y = 0.42 * row / 3.0
                seeds.append((k, x, y))

    base = np.zeros((n, 2), dtype=float)
    base[0] = (0.0, 0.0)
    base[-1] = (1.0, 0.0)
    for k, x, y in seeds:
        base[k] = (x, y)

    rng = np.random.default_rng(164729)
    starts = [base]
    for scale in (0.025, 0.05, 0.085, 0.13, 0.19, 0.27, 0.36, 0.46):
        p = base.copy()
        p[1:-1] += rng.normal(0.0, scale, size=(n - 2, 2))
        p[1:-1] = np.clip(p[1:-1], 0.0, 1.0)
        starts.append(p)

    best_points = base
    best_min = -np.inf

    def try_optimize(start, ftol=2e-10, maxiter=1500):
        """Optimize one feasible distance-packing basin with SLSQP."""
        initial_d2 = squared_distances(
            np.r_[start[variable_indices].ravel(), 0.0]
        )
        initial_t = min(0.05, float(np.sqrt(max(0.0, initial_d2.min()))))
        z0 = np.r_[start[variable_indices].ravel(), initial_t]
        return minimize(
            objective,
            z0,
            method="SLSQP",
            constraints=(
                {"type": "ineq", "fun": lower_constraints},
                {"type": "ineq", "fun": upper_constraints},
            ),
            options={"ftol": ftol, "maxiter": maxiter, "disp": False},
        )

    for start in starts:
        result = try_optimize(start)
        if not result.success or not np.all(np.isfinite(result.x)):
            continue

        candidate = unpack(result.x)
        d2 = squared_distances(result.x)
        if not np.all(np.isfinite(d2)) or d2.max() > 1.0 + 2e-7:
            continue
        candidate_min = float(np.sqrt(max(0.0, d2.min())))
        if candidate_min > best_min:
            best_min = candidate_min
            best_points = candidate

    # Tight deterministic polishing of the best basin.  Small perturbations
    # help SLSQP escape a numerically shallow active-set configuration.
    polish_starts = [best_points]
    for scale in (0.002, 0.006, 0.012):
        p = best_points.copy()
        p[1:-1] += rng.normal(0.0, scale, size=(n - 2, 2))
        p[1:-1] = np.clip(p[1:-1], 0.0, 1.0)
        polish_starts.append(p)

    for start in polish_starts:
        result = try_optimize(start, ftol=2e-12, maxiter=2200)
        if not result.success or not np.all(np.isfinite(result.x)):
            continue
        candidate = unpack(result.x)
        d2 = squared_distances(result.x)
        if not np.all(np.isfinite(d2)) or d2.max() > 1.0 + 5e-8:
            continue
        candidate_min = float(np.sqrt(max(0.0, d2.min())))
        if candidate_min > best_min:
            best_min = candidate_min
            best_points = candidate

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END
