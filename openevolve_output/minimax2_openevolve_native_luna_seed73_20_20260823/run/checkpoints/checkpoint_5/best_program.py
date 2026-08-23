# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Optimize 16 planar points by deterministic annealed maximin search."""
    rng = np.random.default_rng(42)
    n = 16

    # A compact triangular-lattice patch gives a strong starting arrangement.
    lattice = []
    for iy in range(-4, 5):
        for ix in range(-4, 5):
            lattice.append((ix + 0.5 * iy, 0.8660254037844386 * iy))
    lattice = np.asarray(lattice, dtype=float)
    order = np.argsort(np.sum(lattice * lattice, axis=1))
    points = lattice[order[:n]].copy()

    def quality(x):
        delta = x[:, None, :] - x[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        pairs = distances[np.triu_indices(n, 1)]
        return float(np.min(pairs) / np.max(pairs))

    def normalize(x):
        x = x - np.mean(x, axis=0)
        radius = np.max(np.sqrt(np.sum(x * x, axis=1)))
        return x / radius if radius > 1e-14 else x

    current = normalize(points)
    current_value = quality(current)
    best = current.copy()
    best_value = current_value

    # Multiple cooling cycles improve escape from lattice-induced local optima.
    for cycle in range(6):
        for step in range(6500):
            progress = step / 6499.0
            scale = 0.20 * (1.0 - progress) ** 0.72 + 0.003
            temperature = 0.010 * (1.0 - progress) + 0.00012

            trial = current.copy()
            k = int(rng.integers(n))
            trial[k] += rng.normal(0.0, scale, 2)
            trial = normalize(trial)

            value = quality(trial)
            if value >= current_value or rng.random() < np.exp(
                (value - current_value) / temperature
            ):
                current = trial
                current_value = value
                if value > best_value:
                    best = trial.copy()
                    best_value = value

    return normalize(best)


# EVOLVE-BLOCK-END
