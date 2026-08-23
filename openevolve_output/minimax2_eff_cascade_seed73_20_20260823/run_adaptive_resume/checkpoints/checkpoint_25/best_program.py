import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    iu, ju = np.triu_indices(n, 1)

    def normalize(x):
        x = np.asarray(x, dtype=np.float64)
        x = x - np.mean(x, axis=0, keepdims=True)
        q = x[iu] - x[ju]
        diameter2 = float(np.max(np.sum(q * q, axis=1)))
        return x / np.sqrt(max(diameter2, 1e-30))

    def exact(x):
        q = x[iu] - x[ju]
        d = np.sum(q * q, axis=1)
        return float(np.min(d) / max(float(np.max(d)), 1e-30))

    def rings(n0, n1, radius, phase):
        a = 2.0 * np.pi * np.arange(n0, dtype=np.float64) / n0
        b = phase + 2.0 * np.pi * np.arange(n1, dtype=np.float64) / n1
        return np.vstack((
            np.column_stack((np.cos(a), np.sin(a))),
            radius * np.column_stack((np.cos(b), np.sin(b))),
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

            zlo = -beta * d
            mzlo = float(np.max(zlo))
            wl = np.exp(zlo - mzlo)
            wl /= np.sum(wl)
            softlo = -(np.log(np.sum(np.exp(zlo - mzlo))) + mzlo) / beta

            zhi = beta * d
            mzhi = float(np.max(zhi))
            wh = np.exp(zhi - mzhi)
            wh /= np.sum(wh)
            softhi = (np.log(np.sum(np.exp(zhi - mzhi))) + mzhi) / beta

            grad = np.zeros_like(x)
            g = 2.0 * q * (
                wl / max(softlo, 1e-30) - wh / max(softhi, 1e-30)
            )[:, None]
            np.add.at(grad, iu, g)
            np.add.at(grad, ju, -g)
            grad -= np.mean(grad, axis=0, keepdims=True)

            step = 0.022 * (1.0 - it / count) + 0.001
            norm = np.sqrt(np.sum(grad * grad))
            x = normalize(x + step * grad / max(norm, 1e-30))
        return x

    outer = np.arange(10, dtype=np.float64)
    inner = np.arange(5, dtype=np.float64)
    radius = 1.0 / (2.0 * np.cos(np.pi / 10.0))
    incumbent = np.vstack((
        np.zeros((1, 2)),
        radius * np.column_stack((
            np.cos(np.pi / 10.0 + 2.0 * np.pi * inner / 5.0),
            np.sin(np.pi / 10.0 + 2.0 * np.pi * inner / 5.0),
        )),
        np.column_stack((
            np.cos(2.0 * np.pi * outer / 10.0),
            np.sin(2.0 * np.pi * outer / 10.0),
        )),
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
        trajectories = [normalize(seed)]
        if si < 3:
            j = np.arange(n, dtype=np.float64)
            alternating = ((-1.0) ** (j + si))[:, None] * np.array([[1.0, -0.73]])
            wave = np.column_stack((
                np.cos(2.0 * np.pi * (j + si) / n),
                np.sin(2.0 * np.pi * (j + 2 * si + 1) / n),
            ))
            trajectories.extend((
                normalize(seed + 0.004 * alternating),
                normalize(seed + 0.003 * wave),
            ))

        for x in trajectories:
            for beta, count in (
                (4.0, 180), (14.0, 220), (45.0, 280),
                (140.0, 340), (420.0, 420),
            ):
                x = continue_pack(x, beta, count)
            value = exact(x)
            if value > best_score:
                best, best_score = x.copy(), value

    def pair_gradient(x, k, sign):
        """Gradient of sign*d_k, flattened after removing translation."""
        g = np.zeros_like(x)
        v = sign * 2.0 * (x[iu[k]] - x[ju[k]])
        g[iu[k]] += v
        g[ju[k]] -= v
        g -= np.mean(g, axis=0, keepdims=True)
        return g

    def bundle_polish(seed):
        """Polish by cyclic projection onto active min/diameter contact cones."""
        x = normalize(seed)
        score = exact(x)

        for scale in (
            0.012, 0.006, 0.003, 0.0015, 0.00075, 0.000375,
            0.0001875, 0.00009375, 0.000046875, 0.0000234375,
        ):
            while True:
                changed = False
                q = x[iu] - x[ju]
                d = np.sum(q * q, axis=1)
                dmin = float(np.min(d))
                dmax = float(np.max(d))

                # Keep each inequality as an individual half-space.  The
                # bands intentionally include contacts that can become active
                # after diameter normalization during this poll.
                lo = np.flatnonzero(d <= 1.005 * dmin)
                hi = np.flatnonzero(d >= 0.995 * dmax)
                constraints = [
                    pair_gradient(x, k, 1.0).ravel() for k in lo
                ] + [
                    pair_gradient(x, k, -1.0).ravel() for k in hi
                ]

                if constraints:
                    aggregate = np.sum(constraints, axis=0)
                    # Several starts make the cone construction less dependent
                    # on cancellation in the conventional aggregate gradient.
                    starts = [aggregate]
                    starts.extend(constraints)
                    radial = (x - np.mean(x, axis=0, keepdims=True)).ravel()
                    starts.append(radial)

                    for start in starts:
                        v = start.copy()
                        for _ in range(32):
                            for a in constraints:
                                aa = float(np.dot(a, a))
                                if aa > 1e-28:
                                    av = float(np.dot(a, v))
                                    if av < 0.0:
                                        v -= (av / aa) * a

                        vn = np.sqrt(np.dot(v, v))
                        if vn > 1e-30:
                            v = v.reshape(n, 2)
                            v -= np.mean(v, axis=0, keepdims=True)
                            vn = np.sqrt(np.sum(v * v))
                            if vn > 1e-30:
                                # The projection constructs the positive,
                                # jointly feasible orientation only.
                                trial = normalize(x + scale * v / vn)
                                value = exact(trial)
                                if value > score:
                                    x, score, changed = trial, value, True
                                    break

                # Coordinate polls remain useful when a contact graph changes
                # discontinuously and are deliberately tested in both signs.
                if not changed:
                    for p in range(n):
                        for c in range(2):
                            direction = np.zeros_like(x)
                            direction[p, c] = 1.0
                            direction -= np.mean(direction, axis=0, keepdims=True)
                            direction /= np.sqrt(np.sum(direction * direction))
                            for sign in (-1.0, 1.0):
                                trial = normalize(x + sign * scale * direction)
                                value = exact(trial)
                                if value > score:
                                    x, score, changed = trial, value, True

                # The aggregate is a fallback rather than the bundle itself,
                # and therefore both orientations are valid candidates.
                if not changed and constraints:
                    aggregate = np.sum(constraints, axis=0).reshape(n, 2)
                    aggregate -= np.mean(aggregate, axis=0, keepdims=True)
                    an = np.sqrt(np.sum(aggregate * aggregate))
                    if an > 1e-30:
                        for sign in (-1.0, 1.0):
                            trial = normalize(x + sign * scale * aggregate / an)
                            value = exact(trial)
                            if value > score:
                                x, score, changed = trial, value, True

                if not changed:
                    break
        return x, score

    polished, polished_score = bundle_polish(best)
    if polished_score > best_score:
        best, best_score = polished, polished_score

    return np.asarray(normalize(best), dtype=np.float64)