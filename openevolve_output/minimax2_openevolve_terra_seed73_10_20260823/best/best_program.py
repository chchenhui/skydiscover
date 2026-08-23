# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    # The objective is scale invariant, so use a compact hexagonal-lattice
    # configuration as a strong deterministic starting point.  A 19-point
    # hexagonal patch is reduced to its best 16-point subset, then refined.
    def ratio_squared(p: np.ndarray) -> float:
        delta = p[:, None, :] - p[None, :, :]
        dist2 = np.sum(delta * delta, axis=2)
        np.fill_diagonal(dist2, np.inf)
        return float(np.min(dist2) / np.max(dist2[np.isfinite(dist2)]))

    # Axial hexagonal coordinates, converted to Euclidean coordinates.
    axial = [
        (q, r)
        for q in range(-2, 3)
        for r in range(-2, 3)
        if max(abs(q), abs(r), abs(q + r)) <= 2
    ]
    patch = np.array(
        [(q + 0.5 * r, 0.5 * np.sqrt(3.0) * r) for q, r in axial],
        dtype=float,
    )

    # Select the most compact sixteen-point subset of the 19-point patch.
    seed = max(
        (patch[list(indices)] for indices in combinations(range(19), 16)),
        key=ratio_squared,
    )

    # Deterministic simulated annealing improves the lattice seed into a
    # non-lattice packing.  Keep the best point set ever encountered, so this
    # cannot return a result worse than the compact hexagonal initialization.
    rng = np.random.default_rng(20260823)
    best = seed.copy()
    best_value = ratio_squared(best)

    for restart in range(10):
        if restart == 0:
            current = seed.copy()
        else:
            current = seed + rng.normal(scale=0.10, size=seed.shape)

        value = ratio_squared(current)

        for iteration in range(9000):
            progress = iteration / 8999.0
            temperature = 0.004 * (1.0 - progress) ** 2 + 1.0e-6
            step = 0.18 * (1.0 - progress) + 0.002

            trial = current.copy()
            point_index = rng.integers(16)
            trial[point_index] += rng.normal(scale=step, size=2)

            trial_value = ratio_squared(trial)
            if (
                trial_value >= value
                or rng.random() < np.exp((trial_value - value) / temperature)
            ):
                current = trial
                value = trial_value

                if value > best_value:
                    best = current.copy()
                    best_value = value

    # Translation does not affect the metric, but centering keeps coordinates
    # numerically tidy and makes the construction easier to inspect.
    return best - np.mean(best, axis=0)


# EVOLVE-BLOCK-END
