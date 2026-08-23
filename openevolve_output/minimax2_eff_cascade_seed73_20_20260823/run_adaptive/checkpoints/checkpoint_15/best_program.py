# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Deterministically optimize staggered rings and a golden-angle spiral."""
    n = 16
    iu, ju = np.triu_indices(n, 1)

    def normalize(x):
        x = np.asarray(x, dtype=np.float64)
        x = x - np.mean(x, axis=0, keepdims=True)
        q = x[iu] - x[ju]
        return x / np.sqrt(max(float(np.max(np.sum(q * q, axis=1))), 1e-30))

    def exact(x):
        q = x[iu] - x[ju]
        d = np.sum(q * q, axis=1)
        return float(np.min(d) / max(float(np.max(d)), 1e-30))

    def rings(n0, n1, inner_radius, phase):
        a = 2.0 * np.pi * np.arange(n0, dtype=np.float64) / n0
        b = phase + 2.0 * np.pi * np.arange(n1, dtype=np.float64) / n1
        return np.vstack((
            np.column_stack((np.cos(a), np.sin(a))),
            inner_radius * np.column_stack((np.cos(b), np.sin(b))),
        ))

    def spiral():
        k = np.arange(n, dtype=np.float64)
        a = 2.0 * np.pi * k * ((np.sqrt(5.0) - 1.0) / 2.0)
        r = np.sqrt((k + 0.65) / (n + 0.65))
        return np.column_stack((r * np.cos(a), r * np.sin(a)))

    def continue_pack(x, beta, count):
        x = normalize(x)
        for it in range(count):
            q = x[iu] - x[ju]
            d = np.sum(q * q, axis=1)

            lo = -beta * d
            lo -= np.max(lo)
            wl = np.exp(lo)
            wl /= np.sum(wl)
            t = -(np.log(np.sum(wl)) + np.max(-beta * d)) / beta

            hi = beta * d
            hi -= np.max(hi)
            wh = np.exp(hi)
            wh /= np.sum(wh)
            u = (np.log(np.sum(wh)) + np.max(beta * d)) / beta

            grad = np.zeros_like(x)
            g = 2.0 * q * (wl / max(t, 1e-30) -
                           wh / max(u, 1e-30))[:, None]
            np.add.at(grad, iu, g)
            np.add.at(grad, ju, -g)
            grad -= np.mean(grad, axis=0, keepdims=True)

            step = 0.022 * (1.0 - it / count) + 0.001
            grad_norm = np.sqrt(np.sum(grad * grad))
            x = normalize(x + step * grad / max(grad_norm, 1e-30))
        return x

    # The incumbent is retained as an explicit fourth candidate.
    outer = np.arange(10, dtype=np.float64)
    inner = np.arange(5, dtype=np.float64)
    radius = 1.0 / (2.0 * np.cos(np.pi / 10.0))
    incumbent = np.vstack((
        np.zeros((1, 2)),
        radius * np.column_stack((
            np.cos(np.pi / 10.0 + 2.0 * np.pi * inner / 5.0),
            np.sin(np.pi / 10.0 + 2.0 * np.pi * inner / 5.0))),
        np.column_stack((
            np.cos(2.0 * np.pi * outer / 10.0),
            np.sin(2.0 * np.pi * outer / 10.0))),
    ))

    seeds = (
        rings(8, 8, 0.53, np.pi / 8.0),
        rings(7, 9, 0.57, np.pi / 9.0),
        spiral(),
        incumbent,
    )
    best = normalize(incumbent)
    best_score = exact(best)

    for si, seed in enumerate(seeds):
        x = normalize(seed)
        if si < 3:
            # Fixed signed perturbations break exact ring symmetries.
            j = np.arange(n, dtype=np.float64)[:, None]
            x = normalize(x + 0.004 * ((-1.0) ** (j + si)) *
                           np.array([[1.0, -0.73]]))
        for beta, count in ((4.0, 180), (14.0, 220), (45.0, 280),
                            (140.0, 340), (420.0, 420)):
            x = continue_pack(x, beta, count)
        score = exact(x)
        if score > best_score:
            best_score, best = score, x.copy()

    def exact_active_poll(seed):
        """Poll exact squared dispersion along coordinates and active contacts."""
        x = normalize(seed)
        score = exact(x)

        # Continue below the original polishing scales so that nearly active
        # min/max contacts can be resolved without disturbing the incumbent.
        for scale in (
            0.012, 0.006, 0.003, 0.0015, 0.00075, 0.000375,
            0.0001875, 0.00009375, 0.000046875, 0.0000234375,
            0.00001171875, 0.000005859375
        ):
            while True:
                changed = False

                # Repeated coordinate polling lets each accepted move alter
                # the subsequent exact contact sets.
                for p in range(n):
                    for c in range(2):
                        direction = np.zeros_like(x)
                        direction[p, c] = 1.0
                        direction -= np.mean(direction, axis=0)
                        direction /= np.sqrt(np.sum(direction * direction))
                        for sign in (-1.0, 1.0):
                            trial = normalize(x + sign * scale * direction)
                            value = exact(trial)
                            if value > score:
                                x, score = trial, value
                                changed = True

                q = x[iu] - x[ju]
                d = np.sum(q * q, axis=1)
                dmin, dmax = float(np.min(d)), float(np.max(d))
                active = np.zeros_like(x)

                # The two signed terms maximize the minimum while minimizing
                # the diameter, using the current 1% active bands.
                lo = d <= 1.01 * dmin
                hi = d >= 0.99 * dmax
                if np.any(lo):
                    g = 2.0 * q[lo]
                    np.add.at(active, iu[lo], g)
                    np.add.at(active, ju[lo], -g)
                if np.any(hi):
                    g = -2.0 * q[hi]
                    np.add.at(active, iu[hi], g)
                    np.add.at(active, ju[hi], -g)

                active -= np.mean(active, axis=0)
                norm = np.sqrt(np.sum(active * active))
                if norm > 1e-30:
                    active /= norm
                    for sign in (-1.0, 1.0):
                        trial = normalize(x + sign * scale * active)
                        value = exact(trial)
                        if value > score:
                            x, score = trial, value
                            changed = True

                if not changed:
                    break
        return x, score

    # Retain the continuation incumbent and accept only strict exact gains.
    polished, polished_score = exact_active_poll(best)
    if polished_score > best_score:
        best, best_score = polished, polished_score
    return np.asarray(normalize(best), dtype=np.float64)


# EVOLVE-BLOCK-END
