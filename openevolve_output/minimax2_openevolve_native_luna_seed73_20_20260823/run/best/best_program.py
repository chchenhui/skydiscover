# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Construct 16 points by deterministic annealed maximin optimization.

    The initialization is a compact triangular-lattice patch.  A short,
    deterministic simulated-annealing search then maximizes the exact
    minimum-pair-distance/maximum-pair-distance ratio, with translation and
    scale removed before returning the points.
    """
    rng = np.random.default_rng(42)
    n = 16

    # Start from a compact triangular lattice rather than an unconstrained
    # Gaussian cloud, which otherwise tends to contain very close pairs.
    lattice = []
    for iy in range(-4, 5):
        for ix in range(-4, 5):
            lattice.append((ix + 0.5 * iy, 0.8660254037844386 * iy))
    lattice = np.asarray(lattice, dtype=float)
    order = np.argsort(np.sum(lattice * lattice, axis=1))
    points = lattice[order[:n]].copy()

    def quality(x):
        delta = x[:, None, :] - x[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=2))
        tri = dist[np.triu_indices(n, 1)]
        return float(np.min(tri) / np.max(tri))

    # Normalize the initial scale and center to make the numerical search
    # insensitive to the arbitrary lattice spacing.
    points -= np.mean(points, axis=0)
    points /= np.max(np.sqrt(np.sum(points * points, axis=1)))
    best = points.copy()
    best_value = quality(best)
    current = best.copy()
    current_value = best_value

    # Restart every cooling cycle from the best arrangement found so far.
    # This prevents a poor late-cycle state from becoming the starting point
    # of the next cycle while retaining deterministic annealing exploration.
    for cycle in range(12):
        current = best.copy()
        current_value = best_value
        for step in range(6500):
            progress = step / 6499.0
            scale = 0.20 * (1.0 - progress) ** 0.72 + 0.003
            temperature = 0.010 * (1.0 - progress) + 0.00012

            trial = current.copy()
            k = int(rng.integers(n))
            trial[k] += rng.normal(0.0, scale, 2)
            trial -= np.mean(trial, axis=0)
            radius = np.max(np.sqrt(np.sum(trial * trial, axis=1)))
            if radius < 1e-12:
                continue
            trial /= radius

            value = quality(trial)
            accept = value >= current_value
            if not accept:
                accept = rng.random() < np.exp(
                    (value - current_value) / temperature
                )
            if accept:
                current = trial
                current_value = value
                if value > best_value:
                    best = trial.copy()
                    best_value = value

    # Deterministic directional polishing.  Testing several evenly spaced
    # directions avoids the axis-alignment bias of coordinate descent and
    # directly explores the oblique displacements common in dense packings.
    polished = best.copy()
    polished_value = quality(polished)
    # A finer angular stencil improves the final maximin polishing without
    # introducing coordinate-axis bias.
    angles = np.arange(32) * np.pi / 16.0
    directions = np.stack(
        (np.cos(angles), np.sin(angles)),
        axis=1,
    )
    for step_size in (0.025, 0.012, 0.006, 0.003, 0.0015, 0.0007):
        improved = True
        while improved:
            improved = False
            for k in range(n):
                for direction in directions:
                    trial = polished.copy()
                    trial[k] += step_size * direction
                    trial -= np.mean(trial, axis=0)
                    radius = np.max(
                        np.sqrt(np.sum(trial * trial, axis=1))
                    )
                    if radius < 1e-12:
                        continue
                    trial /= radius
                    value = quality(trial)
                    if value > polished_value:
                        polished = trial
                        polished_value = value
                        improved = True
                        if value > best_value:
                            best = trial.copy()
                            best_value = value

    best -= np.mean(best, axis=0)
    best /= np.max(np.sqrt(np.sum(best * best, axis=1)))
    return best


# EVOLVE-BLOCK-END
