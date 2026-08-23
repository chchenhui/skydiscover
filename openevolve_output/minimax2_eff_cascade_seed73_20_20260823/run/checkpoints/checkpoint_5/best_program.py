# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Deterministically optimize a 16-point diameter-normalized packing.

    Several concentric-ring and lattice-like seeds are improved by gradient
    ascent on smooth minimum-distance / maximum-distance approximations.
    The best configuration is selected using the evaluator's exact ratio.
    """
    n = 16
    rng = np.random.default_rng(160216)

    def normalize(x):
        x = x - x.mean(axis=0)
        delta = x[:, None, :] - x[None, :, :]
        diameter = np.sqrt(np.sum(delta * delta, axis=2)).max()
        return x / diameter

    def exact_score(x):
        delta = x[:, None, :] - x[None, :, :]
        d2 = np.sum(delta * delta, axis=2)
        np.fill_diagonal(d2, np.inf)
        return d2.min()

    seeds = []

    # Square-grid seed.
    a = np.arange(4, dtype=float) - 1.5
    seeds.append(np.array([(x, y) for y in a for x in a], dtype=float))

    # Ring seeds: good packings generally need both boundary and interior points.
    for outer in (8, 9, 10, 11):
        inner = n - outer
        for phase in (0.0, np.pi / outer):
            ao = 2.0 * np.pi * np.arange(outer) / outer + phase
            ai = 2.0 * np.pi * np.arange(inner) / inner + 0.37
            seeds.append(np.vstack((
                np.column_stack((np.cos(ao), np.sin(ao))),
                0.47 * np.column_stack((np.cos(ai), np.sin(ai))),
            )))

    # A perturbed hexagonal seed provides a distinct basin of attraction.
    q = np.array([(i + 0.5 * (j & 1), np.sqrt(3.0) * j / 2.0)
                  for j in range(4) for i in range(4)], dtype=float)
    seeds.append(q + 0.08 * rng.standard_normal((n, 2)))

    best = normalize(seeds[0])
    best_value = exact_score(best)

    for seed in seeds:
        x = normalize(seed)
        for it in range(3200):
            # Raising p gradually changes smooth means into close approximations
            # of the active nearest and farthest pair constraints.
            p = 10.0 + 26.0 * it / 3199.0
            delta = x[:, None, :] - x[None, :, :]
            d2 = np.sum(delta * delta, axis=2)
            np.fill_diagonal(d2, 1.0)

            near = d2 ** (-0.5 * p)
            far = d2 ** (0.5 * p)
            np.fill_diagonal(near, 0.0)
            np.fill_diagonal(far, 0.0)
            near /= near.sum()
            far /= far.sum()

            # Gradient of log(smooth-min / smooth-max).
            coeff = (near - far) / d2
            grad = np.sum(coeff[:, :, None] * delta, axis=1)
            grad -= grad.mean(axis=0)

            step = 0.018 * (1.0 - 0.65 * it / 3200.0)
            x = normalize(x + step * grad / (np.sqrt(np.mean(grad * grad)) + 1e-12))

            if it % 80 == 79:
                value = exact_score(x)
                if value > best_value:
                    best_value = value
                    best = x.copy()

    # Deterministic exact-objective annealing followed by greedy polishing.
    # Occasional downhill moves help reorganize several simultaneously active
    # nearest-neighbor constraints instead of freezing at the first local peak.
    x = best.copy()
    value = best_value
    for it in range(90000):
        frac = it / 89999.0
        y = x.copy()
        scale = 0.020 * (1.0 - frac) + 0.00012
        if it % 4 == 0:
            ids = rng.choice(n, size=2, replace=False)
            y[ids] += scale * rng.standard_normal((2, 2))
        else:
            k = int(rng.integers(n))
            y[k] += scale * rng.standard_normal(2)
        y = normalize(y)
        v = exact_score(y)
        temperature = 7.0e-5 * (1.0 - frac) ** 2 + 2.0e-8
        if v >= value or rng.random() < np.exp((v - value) / temperature):
            x, value = y, v
            if value > best_value:
                best, best_value = x.copy(), value

    x = best.copy()
    value = best_value
    for it in range(70000):
        y = x.copy()
        frac = it / 69999.0
        scale = 0.012 * (1.0 - frac) + 0.00008
        if it % 3 == 0:
            ids = rng.choice(n, size=2, replace=False)
            y[ids] += scale * rng.standard_normal((2, 2))
        else:
            k = int(rng.integers(n))
            y[k] += scale * rng.standard_normal(2)
        y = normalize(y)
        v = exact_score(y)
        if v > value:
            x, value = y, v
            if value > best_value:
                best, best_value = x.copy(), value

    return best


# EVOLVE-BLOCK-END
