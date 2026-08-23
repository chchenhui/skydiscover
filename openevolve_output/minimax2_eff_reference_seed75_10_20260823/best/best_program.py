import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Deterministic lattice annealing followed by active-contact refinement."""
    rng = np.random.default_rng(271828)

    q = np.arange(4, dtype=float) - 1.5
    points = np.array([(x, y) for x in q for y in q], dtype=float)
    points += rng.normal(0.0, 0.035, points.shape)

    iu = np.triu_indices(16, 1)

    def score(x):
        y = x - x.mean(axis=0)
        delta = y[:, None, :] - y[None, :, :]
        ds2 = np.sum(delta * delta, axis=2)
        vals = ds2[iu]
        return np.sqrt(vals.min() / vals.max())

    def pair_data(x):
        y = x - x.mean(axis=0)
        delta = y[:, None, :] - y[None, :, :]
        ds2 = np.sum(delta * delta, axis=2)
        vals = ds2[iu]
        imin = int(np.argmin(vals))
        imax = int(np.argmax(vals))
        return y, ds2, (iu[0][imin], iu[1][imin]), (iu[0][imax], iu[1][imax])

    best = points.copy()
    best_score = score(best)
    current = best.copy()
    current_score = best_score

    # Original broad basin search, retained as a dependable incumbent generator.
    for k in range(180000):
        frac = k / 180000.0
        temperature = 0.018 * (1.0 - frac) ** 1.7 + 1.0e-5
        step = 0.16 * (1.0 - frac) ** 0.8 + 0.002

        i = int(rng.integers(16))
        trial = current.copy()
        trial[i] += rng.normal(0.0, step, 2)
        trial_score = score(trial)
        gain = trial_score - current_score

        if gain > 0.0 or rng.random() < np.exp(gain / temperature):
            current = trial
            current_score = trial_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score

        if k and k % 30000 == 0:
            current = best + rng.normal(0.0, 0.025, best.shape)
            current_score = score(current)

    # Active-contact refinement: shortest contacts repel and diameter contacts
    # attract.  Candidate acceptance always protects the annealed incumbent.
    current = best.copy()
    current_score = best_score
    for k in range(130000):
        frac = k / 130000.0
        y, ds2, min_pair, max_pair = pair_data(current)
        dmin2 = ds2[min_pair]
        dmax2 = ds2[max_pair]
        dmin = np.sqrt(dmin2)
        dmax = np.sqrt(dmax2)

        force = np.zeros((16, 2), dtype=float)

        # Treat near-tied contacts together, avoiding arbitrary choices among
        # equivalent shortest and longest edges.
        for a, b in zip(iu[0], iu[1]):
            dd2 = ds2[a, b]
            vec = y[a] - y[b]
            dd = np.sqrt(dd2)
            if dd == 0.0:
                vec = rng.normal(0.0, 1.0, 2)
                dd = np.linalg.norm(vec)
            unit = vec / dd

            if dd2 <= dmin2 * 1.075:
                w = 1.075 - dd2 / dmin2
                force[a] += w * unit
                force[b] -= w * unit
            if dd2 >= dmax2 * 0.935:
                w = dd2 / dmax2 - 0.935
                force[a] -= w * unit
                force[b] += w * unit

        norms = np.sqrt(np.sum(force * force, axis=1))
        largest = norms.max()
        if largest > 0.0:
            force /= largest

        active_step = 0.028 * (1.0 - frac) ** 1.35 + 0.00018
        noise_step = 0.010 * (1.0 - frac) ** 1.7 + 0.00003

        trial = current + active_step * force
        # A small selected-point perturbation lets the active contact graph
        # change instead of becoming trapped at a fixed set of ties.
        chosen = int(rng.integers(16))
        trial[chosen] += rng.normal(0.0, noise_step, 2)
        trial -= trial.mean(axis=0)

        trial_score = score(trial)
        gain = trial_score - current_score
        temperature = 0.0018 * (1.0 - frac) ** 2.2 + 1.0e-8

        if gain > 0.0 or rng.random() < np.exp(gain / temperature):
            current = trial
            current_score = trial_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score

        # Periodically restart only the working state; best remains monotone.
        if k and k % 26000 == 0:
            current = best + rng.normal(0.0, 0.0045, best.shape)
            current_score = score(current)

    """Apply deterministic compass-and-contact stencil search after annealing."""
    current = best.copy()
    target = 0.0725472451305
    compass = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
         [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
        dtype=float,
    )
    compass /= np.maximum(
        np.linalg.norm(compass, axis=1, keepdims=True), 1.0
    )
    steps = (0.012, 0.006, 0.003, 0.0015, 0.00075, 0.000375,
             0.0001875, 0.00009375, 0.000046875)

    def raw_ratio_squared(x):
        z = x[:, None, :] - x[None, :, :]
        d2 = np.sum(z * z, axis=2)[iu]
        return float(d2.min() / d2.max())

    for step in steps:
        while True:
            base_value = raw_ratio_squared(current)
            best_value = base_value
            best_trial = None

            y, ds2, min_pair, max_pair = pair_data(current)
            shortest = y[min_pair[0]] - y[min_pair[1]]
            longest = y[max_pair[0]] - y[max_pair[1]]
            sn = np.linalg.norm(shortest)
            ln = np.linalg.norm(longest)
            active = []
            if sn > 0.0:
                active.append(shortest / sn)
                active.append(-shortest / sn)
            if ln > 0.0:
                active.append(longest / ln)
                active.append(-longest / ln)

            for i in range(16):
                directions = list(compass)
                if i == min_pair[0]:
                    directions.append(shortest / sn if sn > 0.0 else np.zeros(2))
                elif i == min_pair[1]:
                    directions.append(-shortest / sn if sn > 0.0 else np.zeros(2))
                else:
                    directions.append(np.zeros(2))
                if i == max_pair[0]:
                    directions.append(-longest / ln if ln > 0.0 else np.zeros(2))
                elif i == max_pair[1]:
                    directions.append(longest / ln if ln > 0.0 else np.zeros(2))
                else:
                    directions.append(np.zeros(2))

                for direction in directions:
                    if not np.any(direction):
                        continue
                    trial = current.copy()
                    trial[i] += step * direction
                    trial -= trial.mean(axis=0)
                    value = raw_ratio_squared(trial)
                    if value > best_value and value > target:
                        best_value = value
                        best_trial = trial

            if best_trial is None:
                break
            current = best_trial
            if best_value > raw_ratio_squared(best):
                best = current.copy()

    best -= best.mean(axis=0)
    delta = best[:, None, :] - best[None, :, :]
    dmax = np.sqrt(np.sum(delta * delta, axis=2)).max()
    return best / dmax