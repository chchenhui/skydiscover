import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    rng = np.random.default_rng(160216)

    def normalize(x):
        x = np.asarray(x, dtype=float).copy()
        x -= x.mean(axis=0)
        z = x[:, None, :] - x[None, :, :]
        diam = np.sqrt(np.sum(z * z, axis=2)).max()
        return x / diam

    def score(x):
        z = x[:, None, :] - x[None, :, :]
        d2 = np.sum(z * z, axis=2)
        np.fill_diagonal(d2, np.inf)
        return float(d2.min())

    seeds = []
    a = np.arange(4.0) - 1.5
    seeds.append(np.array([(u, v) for v in a for u in a], float))
    for outer in (8, 9, 10, 11):
        inner = n - outer
        for phase in (0.0, np.pi / outer):
            ao = 2.0 * np.pi * np.arange(outer) / outer + phase
            ai = 2.0 * np.pi * np.arange(inner) / inner + 0.37
            seeds.append(np.vstack((
                np.column_stack((np.cos(ao), np.sin(ao))),
                .47 * np.column_stack((np.cos(ai), np.sin(ai)))
            )))
    q = np.array([(i + .5 * (j & 1), np.sqrt(3.0) * j / 2.0)
                  for j in range(4) for i in range(4)], float)
    seeds.append(q + .08 * rng.standard_normal((n, 2)))

    best = normalize(seeds[0])
    bestv = score(best)

    # Smooth continuation supplies a good basin for the exact constrained phase.
    for seed in seeds:
        x = normalize(seed)
        for it in range(3200):
            p = 10.0 + 26.0 * it / 3199.0
            z = x[:, None, :] - x[None, :, :]
            d2 = np.sum(z * z, axis=2)
            np.fill_diagonal(d2, 1.0)
            near = d2 ** (-.5 * p)
            far = d2 ** (.5 * p)
            np.fill_diagonal(near, 0.0)
            np.fill_diagonal(far, 0.0)
            near /= near.sum()
            far /= far.sum()
            g = np.sum(((near - far) / d2)[:, :, None] * z, axis=1)
            g -= g.mean(axis=0)
            step = .018 * (1.0 - .65 * it / 3200.0)
            x = normalize(x + step * g /
                          (np.sqrt(np.mean(g * g)) + 1.e-12))
            if it % 80 == 79:
                v = score(x)
                if v > bestv:
                    best, bestv = x.copy(), v

    # A modest exact anneal remains a NumPy-only fallback if SciPy is absent.
    x, v = best.copy(), bestv
    for it in range(90000):
        f = it / 89999.0
        y = x.copy()
        h = .020 * (1.0 - f) + .00012
        if it % 4 == 0:
            ids = rng.choice(n, 2, replace=False)
            y[ids] += h * rng.standard_normal((2, 2))
        else:
            y[int(rng.integers(n))] += h * rng.standard_normal(2)
        y = normalize(y)
        w = score(y)
        temp = 7.e-5 * (1.0 - f) ** 2 + 2.e-8
        if w >= v or rng.random() < np.exp((w - v) / temp):
            x, v = y, w
            if w > bestv:
                best, bestv = x.copy(), w

    # Smooth epigraph constraints:
    # d_ij^2 - t >= 0 and 1 - d_ij^2 >= 0 for every i < j.
    # This optional stage is deterministic and its result is accepted only
    # through the evaluator's exact metric.
    try:
        from scipy.optimize import minimize

        ii, jj = np.triu_indices(n, 1)
        m = len(ii)

        def fun(y):
            return -y[-1]

        def fun_jac(y):
            out = np.zeros_like(y)
            out[-1] = -1.0
            return out

        def con_fun(y):
            p = y[:-1].reshape(n, 2)
            d = p[ii] - p[jj]
            d2 = np.einsum("ij,ij->i", d, d)
            return np.concatenate((d2 - y[-1], 1.0 - d2))

        def con_jac(y):
            p = y[:-1].reshape(n, 2)
            d = p[ii] - p[jj]
            J = np.zeros((2 * m, 2 * n + 1))
            rows = np.arange(m)
            J[rows, 2 * ii] = 2.0 * d[:, 0]
            J[rows, 2 * ii + 1] = 2.0 * d[:, 1]
            J[rows, 2 * jj] = -2.0 * d[:, 0]
            J[rows, 2 * jj + 1] = -2.0 * d[:, 1]
            J[rows, -1] = -1.0
            J[m + rows, :2 * n] = -J[rows, :2 * n]
            return J

        starts = [best]
        probe_rng = np.random.default_rng(7160216)
        for scale in (0.0015, 0.0040, 0.0080):
            starts.append(normalize(
                best + scale * probe_rng.standard_normal((n, 2))
            ))

        for start in starts:
            y0 = np.r_[start.ravel(), score(start)]
            res = minimize(
                fun, y0, jac=fun_jac,
                constraints={"type": "ineq", "fun": con_fun, "jac": con_jac},
                method="SLSQP",
                options={"maxiter": 1800, "ftol": 1.e-13, "disp": False}
            )
            if res.x.size == 2 * n + 1 and np.all(np.isfinite(res.x)):
                candidate = normalize(res.x[:-1].reshape(n, 2))
                cv = score(candidate)
                if cv > bestv:
                    best, bestv = candidate, cv
    except Exception:
        pass

    return best