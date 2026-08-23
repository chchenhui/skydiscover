# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Relax a perturbed pentagon--decagon seed under diameter continuation.

    Starting from the strong 1+5+10 construction, this deterministic
    penalty relaxation gradually lowers an imposed diameter while repelling
    pairs closer than unit distance.  The best actual minimum/maximum ratio
    encountered is retained, so the original symmetric seed remains an
    available fallback.
    """
    inner_angles = 2.0 * np.pi * np.arange(5, dtype=np.float64) / 5.0
    outer_angles = np.pi / 10.0 + 2.0 * np.pi * np.arange(10, dtype=np.float64) / 10.0
    outer_radius = 2.0 * np.cos(np.pi / 10.0)

    points = np.vstack((
        np.array([[0.0, 0.0]]),
        np.column_stack((np.cos(inner_angles), np.sin(inner_angles))),
        outer_radius * np.column_stack((np.cos(outer_angles), np.sin(outer_angles))),
    )).astype(np.float64)

    ii, jj = np.triu_indices(16, 1)
    delta = points[ii] - points[jj]
    dd = np.sqrt(np.sum(delta * delta, axis=1))
    best = points.copy()
    best_ratio = dd.min() / dd.max()

    # Break the ring symmetry reproducibly; the unperturbed packing above is
    # retained as best until a genuine improvement is measured.
    k = np.arange(16, dtype=np.float64)
    points += 0.018 * np.column_stack((
        np.sin(2.71 * k + 0.4),
        np.cos(1.83 * k + 0.9),
    ))
    points -= points.mean(axis=0)

    steps = 5000
    for it in range(steps):
        target = 3.80 - 0.255 * it / (steps - 1)
        delta = points[ii] - points[jj]
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        unit = delta / np.maximum(dist[:, None], 1.0e-12)

        # Unit-separation repulsion and a stronger soft diameter wall.
        coeff = np.where(dist < 1.0, -2.0 * (1.0 - dist), 0.0)
        coeff += np.where(dist > target, 8.0 * (dist - target), 0.0)
        force = coeff[:, None] * unit

        grad = np.zeros_like(points)
        np.add.at(grad, ii, force)
        np.add.at(grad, jj, -force)
        points -= 0.018 * grad
        points -= points.mean(axis=0)

        if it % 10 == 9:
            delta = points[ii] - points[jj]
            dist = np.sqrt(np.sum(delta * delta, axis=1))
            ratio = dist.min() / dist.max()
            if ratio > best_ratio:
                best_ratio = ratio
                best = points.copy()

    return best


# EVOLVE-BLOCK-END
