# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Optimize a deterministic 16-point packing with unit minimum separation.

    The construction starts from a compact five-row triangular-lattice patch and
    uses deterministic constrained optimization to minimize its diameter while
    enforcing every pairwise distance to be at least one.  The final rescaling
    adds a tiny numerical safety margin without changing the score materially.
    """
    from scipy.optimize import minimize

    # A compact 3-3-4-3-3 triangular-lattice seed.  It is substantially better
    # than unconstrained Gaussian samples even before the continuous refinement.
    h = np.sqrt(3.0) / 2.0
    seed = np.array([
        [0.5,  2.0 * h], [1.5,  2.0 * h], [2.5,  2.0 * h],
        [0.0,  1.0 * h], [1.0,  1.0 * h], [2.0,  1.0 * h],
        [-0.5, 0.0],     [0.5,  0.0],     [1.5,  0.0], [2.5, 0.0],
        [0.0, -1.0 * h], [1.0, -1.0 * h], [2.0, -1.0 * h],
        [0.5, -2.0 * h], [1.5, -2.0 * h], [2.5, -2.0 * h],
    ], dtype=float)

    # Remove translation freedom: point zero remains fixed at the origin.
    seed -= seed[0]
    iu = np.triu_indices(16, 1)

    def unpack(z):
        points = np.empty((16, 2), dtype=float)
        points[0] = 0.0
        points[1:] = z[:-1].reshape(15, 2)
        return points

    def squared_distances(points):
        delta = points[iu[0]] - points[iu[1]]
        return np.einsum("ij,ij->i", delta, delta)

    def objective(z):
        return z[-1]

    def constraints(z):
        distances2 = squared_distances(unpack(z))
        diameter = z[-1]
        # First block enforces d_min >= 1; second enforces d_max <= diameter.
        return np.concatenate((distances2 - 1.0, diameter * diameter - distances2))

    rng = np.random.default_rng(20260823)
    best_points = seed.copy()
    best_diameter = np.sqrt(squared_distances(seed).max())

    # Several deterministic perturbations help escape the symmetric lattice
    # stationary point while retaining a compact, highly feasible starting shape.
    starts = [seed]
    for _ in range(10):
        trial = seed.copy()
        trial[1:] += rng.normal(scale=0.09, size=(15, 2))
        starts.append(trial)

    bounds = [(-5.0, 5.0)] * 30 + [(1.0, 5.0)]
    for start in starts:
        start_diameter = np.sqrt(squared_distances(start).max()) + 0.15
        x0 = np.concatenate((start[1:].ravel(), [start_diameter]))
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 1800, "ftol": 1e-11, "disp": False},
        )

        if result.success:
            candidate = unpack(result.x)
            candidate_distances2 = squared_distances(candidate)
            if candidate_distances2.min() >= 1.0 - 1e-7:
                candidate_diameter = np.sqrt(candidate_distances2.max())
                if candidate_diameter < best_diameter:
                    best_points = candidate
                    best_diameter = candidate_diameter

    # Exact uniform scaling preserves dmin/dmax and guards against tiny solver
    # feasibility violations caused by floating-point termination tolerances.
    best_points -= best_points.mean(axis=0)
    min_distance = np.sqrt(squared_distances(best_points).min())
    best_points *= (1.0 + 1e-9) / min_distance
    return best_points


# EVOLVE-BLOCK-END
