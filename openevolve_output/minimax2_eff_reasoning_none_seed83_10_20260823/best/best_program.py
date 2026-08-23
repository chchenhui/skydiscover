import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Refine compact triangular-lattice seeds under unit-separation constraints.

    The optimization minimizes a common diameter bound for all 120 pairs,
    beginning from a 16-point clipped hexagonal lattice and several fixed
    deterministic perturbations.  The returned candidate is normalized so
    its smallest pairwise distance is exactly one.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        # This compact clipped triangular lattice has dmin=1 and dmax=sqrt(13),
        # already much better than the square-grid fallback.
        h = np.sqrt(3.0) / 2.0
        return np.array(
            [
                [1.0, 0.0], [2.0, 0.0],
                [0.5, h], [1.5, h], [2.5, h], [3.5, h],
                [0.0, 2.0 * h], [1.0, 2.0 * h],
                [2.0, 2.0 * h], [3.0, 2.0 * h],
                [0.5, 3.0 * h], [1.5, 3.0 * h],
                [2.5, 3.0 * h], [3.5, 3.0 * h],
                [1.0, 4.0 * h], [2.0, 4.0 * h],
            ],
            dtype=float,
        )

    h = np.sqrt(3.0) / 2.0
    base = np.array(
        [
            [1.0, 0.0], [2.0, 0.0],
            [0.5, h], [1.5, h], [2.5, h], [3.5, h],
            [0.0, 2.0 * h], [1.0, 2.0 * h],
            [2.0, 2.0 * h], [3.0, 2.0 * h],
            [0.5, 3.0 * h], [1.5, 3.0 * h],
            [2.5, 3.0 * h], [3.5, 3.0 * h],
            [1.0, 4.0 * h], [2.0, 4.0 * h],
        ],
        dtype=float,
    )
    pair_i, pair_j = np.triu_indices(16, 1)

    def distances_squared(points: np.ndarray) -> np.ndarray:
        delta = points[pair_i] - points[pair_j]
        return np.einsum("ij,ij->i", delta, delta)

    def normalize(points: np.ndarray) -> np.ndarray:
        points = points - points.mean(axis=0)
        return points / np.sqrt(distances_squared(points).min())

    def constraints(z: np.ndarray) -> np.ndarray:
        points = z[:-1].reshape(16, 2)
        diameter = z[-1]
        dsq = distances_squared(points)
        # Unit separation and the common upper diameter bound.
        return np.concatenate((dsq - 1.0, diameter * diameter - dsq))

    best = normalize(base)
    best_ratio = distances_squared(best).min() / distances_squared(best).max()

    rng = np.random.default_rng(160216)
    starts = [best]

    # Use multiple deterministic perturbation scales.  Small perturbations
    # refine the triangular basin, while larger ones can reach alternate
    # contact-graph configurations.
    for scale, count in ((0.035, 4), (0.075, 8), (0.13, 8), (0.22, 6)):
        for _ in range(count):
            trial = base + rng.normal(scale=scale, size=(16, 2))
            starts.append(normalize(trial))

    for start in starts:
        initial_diameter = np.sqrt(distances_squared(start).max())
        result = minimize(
            fun=lambda z: z[-1],
            x0=np.concatenate((start.ravel(), [initial_diameter])),
            method="SLSQP",
            bounds=[(None, None)] * 32 + [(1.0, 10.0)],
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 1800, "ftol": 1e-12, "disp": False},
        )

        candidate = normalize(result.x[:-1].reshape(16, 2))
        dsq = distances_squared(candidate)
        ratio = dsq.min() / dsq.max()
        if ratio > best_ratio:
            best = candidate
            best_ratio = ratio

    return best