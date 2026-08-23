import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Run a normalized topology-diverse elitist search, followed by greedy
    closest-contact expansion and farthest-contact contraction refinement.
    """
    rng = np.random.default_rng(389117)

    def normalize(batch: np.ndarray) -> np.ndarray:
        batch = batch - batch.mean(axis=1, keepdims=True)
        delta = batch[:, :, None, :] - batch[:, None, :, :]
        d2 = np.sum(delta * delta, axis=-1)
        diameter = np.sqrt(np.max(d2, axis=(1, 2)))
        return batch / diameter[:, None, None]

    def scores(batch: np.ndarray) -> np.ndarray:
        delta = batch[:, :, None, :] - batch[:, None, :, :]
        d2 = np.sum(delta * delta, axis=-1)
        ind = np.arange(16)
        d2[:, ind, ind] = np.inf
        minimum = np.min(d2, axis=(1, 2))
        d2[:, ind, ind] = 0.0
        maximum = np.max(d2, axis=(1, 2))
        return minimum / maximum

    # Exact 11-outer / 5-inner annular construction, with minimum distance 1
    # before the harmless diameter normalization.
    a = np.pi / 11.0
    r = 1.0 / (2.0 * np.sin(2.0 * a))
    R = r * np.cos(a) + np.sqrt(1.0 - (r * np.sin(a)) ** 2)

    outer_angles = 2.0 * a * np.arange(11)
    inner_angles = a + 4.0 * a * np.arange(5)
    outer = np.column_stack((R * np.cos(outer_angles), R * np.sin(outer_angles)))
    inner = np.column_stack((r * np.cos(inner_angles), r * np.sin(inner_angles)))
    base = np.vstack((outer, inner))
    base = normalize(base[None, :, :])[0]

    # Three initial topologies: annular, square-grid, and unstructured cloud.
    population_size = 160
    population = np.empty((population_size, 16, 2), dtype=float)

    population[:80] = base + rng.normal(0.0, 0.060, size=(80, 16, 2))
    population[0] = base

    gx, gy = np.meshgrid(np.arange(4, dtype=float), np.arange(4, dtype=float))
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    grid = normalize(grid[None, :, :])[0]
    population[80:120] = grid + rng.normal(0.0, 0.080, size=(40, 16, 2))

    population[120:] = rng.normal(0.0, 1.0, size=(40, 16, 2))
    population = normalize(population)

    best = base.copy()
    best_score = scores(best[None, :, :])[0]

    # Elitist structural exploration over the three initialized topologies.
    for generation in range(180):
        value = scores(population)
        generation_best = int(np.argmax(value))
        if value[generation_best] > best_score:
            best_score = value[generation_best]
            best = population[generation_best].copy()

        order = np.argsort(value)[::-1]
        elites = population[order[:32]]
        new_population = np.empty_like(population)
        new_population[:16] = elites[:16]

        count = population_size - 16
        parents = elites[rng.integers(0, 32, size=count)]
        children = parents.copy()

        # Differential candidates explicitly use three independently selected
        # elite parents, preserving a route between distinct good topologies.
        differential = rng.random(count) < 0.30
        differential_rows = np.flatnonzero(differential)
        if differential_rows.size:
            first = elites[rng.integers(0, 32, size=differential_rows.size)]
            second = elites[rng.integers(0, 32, size=differential_rows.size)]
            third = elites[rng.integers(0, 32, size=differential_rows.size)]
            children[differential_rows] = first + 0.50 * (second - third)

        # Every offspring receives exactly one, two, or three point mutations.
        sigma = 0.065 * (0.975 ** generation) + 0.0015
        mutation_count = rng.integers(1, 4, size=count)
        rows = np.arange(count)
        for mutation in range(3):
            changed_rows = rows[mutation_count > mutation]
            point_ids = rng.integers(0, 16, size=changed_rows.size)
            children[changed_rows, point_ids] += rng.normal(
                0.0, sigma, size=(changed_rows.size, 2)
            )

        broad = rng.random(count) < 0.10
        children[broad] += rng.normal(
            0.0, 0.35 * sigma, size=(np.count_nonzero(broad), 16, 2)
        )

        new_population[16:] = children
        population = normalize(new_population)

    value = scores(population)
    final_best = int(np.argmax(value))
    if value[final_best] > best_score:
        best_score = value[final_best]
        best = population[final_best].copy()

    # Greedy active-contact refinement of the retained global best.
    for step in range(240):
        batch_size = 112
        candidates = np.repeat(best[None, :, :], batch_size, axis=0)
        sigma = 0.035 * (0.985 ** step) + 0.0008
        rows = np.arange(batch_size)

        # Use the same one-to-three individual-point neighborhood during the
        # local phase rather than allowing unperturbed duplicate candidates.
        mutation_count = rng.integers(1, 4, size=batch_size)
        for mutation in range(3):
            changed_rows = rows[mutation_count > mutation]
            point_ids = rng.integers(0, 16, size=changed_rows.size)
            candidates[changed_rows, point_ids] += rng.normal(
                0.0, sigma, size=(changed_rows.size, 2)
            )

        delta = best[:, None, :] - best[None, :, :]
        d2 = np.sum(delta * delta, axis=-1)

        near_d2 = d2.copy()
        np.fill_diagonal(near_d2, np.inf)
        near_i, near_j = divmod(int(np.argmin(near_d2)), 16)
        near_direction = best[near_i] - best[near_j]
        near_direction /= np.linalg.norm(near_direction)

        far_d2 = d2.copy()
        np.fill_diagonal(far_d2, 0.0)
        far_i, far_j = divmod(int(np.argmax(far_d2)), 16)
        far_direction = best[far_i] - best[far_j]
        far_direction /= np.linalg.norm(far_direction)

        push_rows = rows[::3]
        candidates[push_rows, near_i] += sigma * near_direction
        candidates[push_rows, near_j] -= sigma * near_direction

        pull_rows = rows[1::3]
        candidates[pull_rows, far_i] -= sigma * far_direction
        candidates[pull_rows, far_j] += sigma * far_direction

        broad_rows = rows[2::11]
        candidates[broad_rows] += rng.normal(
            0.0, 0.25 * sigma, size=(broad_rows.size, 16, 2)
        )

        candidates = normalize(candidates)
        value = scores(candidates)
        candidate_best = int(np.argmax(value))
        if value[candidate_best] > best_score:
            best_score = value[candidate_best]
            best = candidates[candidate_best].copy()

    return best