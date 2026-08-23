import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Anneal a triangular seed, then polish active min/max distance contacts."""
    rng = np.random.RandomState(16016)
    rt3 = np.sqrt(3.0)
    removed = {(2, 0), (0, -2), (-2, 2)}
    axial = [
        (q, r)
        for q in range(-2, 3)
        for r in range(-2, 3)
        if max(abs(q), abs(r), abs(q + r)) <= 2 and (q, r) not in removed
    ]
    best = np.asarray(
        [[q + 0.5 * r, 0.5 * rt3 * r] for q, r in axial], dtype=float
    )

    def objective(x):
        d = x[:, None, :] - x[None, :, :]
        v = np.sum(d * d, axis=2)[np.triu_indices(16, 1)]
        return float(np.min(v) / np.max(v))

    current = best.copy()
    current_value = objective(current)
    best_value = current_value

    for iteration in range(120000):
        fraction = iteration / 120000.0
        step = 0.34 * (1.0 - fraction) ** 1.7 + 0.004
        candidate = current.copy()
        if iteration % 13 == 0:
            noise = rng.normal(size=(16, 2))
            noise -= noise.mean(axis=0, keepdims=True)
            candidate += 0.42 * step * noise
        else:
            k = int(rng.randint(16))
            candidate[k] += rng.normal(size=2) * step
        value = objective(candidate)
        temperature = 0.0007 * (1.0 - fraction) + 0.000002
        delta = value - current_value
        if delta > 0.0 or rng.rand() < np.exp(delta / temperature):
            current = candidate
            current_value = value
            if value > best_value:
                best = candidate.copy()
                best_value = value

    # Keep an unconditional fallback in case active-contact polishing
    # temporarily visits a less favorable basin.
    annealed_best = best.copy()

    def distances(x):
        delta = x[:, None, :] - x[None, :, :]
        matrix = np.sum(delta * delta, axis=2)
        return matrix[np.triu_indices(16, 1)]

    # Active-contact force polishing.  Near-minimum contacts are pushed
    # apart, while near-diameter contacts are pulled toward one another.
    # Use a wider initial collective move, then geometrically refine below
    # the previous polishing cutoff.  The strict objective test and the
    # annealed fallback make larger exploratory moves safe.
    for step in (
        0.16, 0.08, 0.04, 0.02, 0.01, 0.005,
        0.0025, 0.00125, 0.000625, 0.0003125
    ):
        improved = True
        while improved:
            improved = False
            delta = best[:, None, :] - best[None, :, :]
            matrix = np.sum(delta * delta, axis=2)
            upper = matrix[np.triu_indices(16, 1)]
            dmin = upper.min()
            dmax = upper.max()
            # Include a slightly broader active shell so that contacts
            # which become limiting after a force move participate in the
            # next deterministic refinement.
            min_pairs = np.argwhere(
                np.triu(matrix <= dmin * 1.04, 1)
            )
            max_pairs = np.argwhere(
                np.triu(matrix >= dmax * 0.96, 1)
            )

            force = np.zeros_like(best)
            for i, j in min_pairs:
                v = best[i] - best[j]
                norm = np.linalg.norm(v)
                if norm > 1.e-12:
                    v /= norm
                    force[i] += v
                    force[j] -= v
            for i, j in max_pairs:
                v = best[j] - best[i]
                norm = np.linalg.norm(v)
                if norm > 1.e-12:
                    v /= norm
                    force[i] += v
                    force[j] -= v

            norm = np.linalg.norm(force, axis=1).max()
            if norm > 1.e-12:
                force /= norm
                for sign in (1.0, -1.0):
                    candidate = best + sign * step * force
                    value = objective(candidate)
                    if value > best_value:
                        best, best_value = candidate, value
                        improved = True
                        break

    # Finish with deterministic angular coordinate searches, including
    # non-axis-aligned directions and scales below the prior cutoff.
    angles = np.arange(24, dtype=float) * (2.0 * np.pi / 24.0)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    # Angular searches finish every active-contact refinement scale and
    # retain sub-cutoff motions that can resolve non-axis-aligned contacts.
    for step in (
        0.006, 0.003, 0.0015, 0.00075,
        0.000375, 0.0001875, 0.00009375
    ):
        improved = True
        while improved:
            improved = False
            for k in range(16):
                for direction in directions:
                    candidate = best.copy()
                    candidate[k] += step * direction
                    value = objective(candidate)
                    if value > best_value:
                        best, best_value = candidate, value
                        improved = True

    if objective(annealed_best) > best_value:
        best = annealed_best
    best -= best.mean(axis=0)
    return np.asarray(best, dtype=float).reshape(16, 2)