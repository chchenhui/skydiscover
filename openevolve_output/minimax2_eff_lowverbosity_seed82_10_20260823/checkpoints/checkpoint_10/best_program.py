import numpy as np
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """Optimize diameter-normalized separation using lattice, ring, and greedy-maximin starts."""
    rt3 = np.sqrt(3.0)
    deleted = {(2, 0), (-2, 2), (0, 2)}
    pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q + r)) <= 2 and (q, r) not in deleted:
                pts.append((q + 0.5 * r, 0.5 * rt3 * r))
    base = np.array(pts, float)
    pairs = np.array([(i, j) for i in range(16) for j in range(i)], int)

    def normalize(x):
        x = x - x.mean(0)
        d = x[:, None] - x[None, :]
        return x / np.sqrt((d * d).sum(2).max())

    def ratio(x):
        d = x[pairs[:, 0]] - x[pairs[:, 1]]
        ds = (d * d).sum(1)
        return np.sqrt(ds.min() / ds.max())

    base = normalize(base)
    best = base.copy()
    bestv = ratio(best)

    def fun(z):
        return -z[-1]

    def jacfun(z):
        g = np.zeros_like(z)
        g[-1] = -1.0
        return g

    def con(z):
        x, t = z[:-1].reshape(16, 2), z[-1]
        v = x[pairs[:, 0]] - x[pairs[:, 1]]
        ds = (v * v).sum(1)
        return np.r_[ds - t * t, 1.0 - ds]

    def jaccon(z):
        x, t = z[:-1].reshape(16, 2), z[-1]
        v = x[pairs[:, 0]] - x[pairs[:, 1]]
        m = len(pairs)
        j = np.zeros((2 * m, 33))
        for k, (a, b) in enumerate(pairs):
            j[k, 2*a:2*a+2] = 2*v[k]
            j[k, 2*b:2*b+2] = -2*v[k]
            j[k, -1] = -2*t
            j[m+k, 2*a:2*a+2] = -2*v[k]
            j[m+k, 2*b:2*b+2] = 2*v[k]
        return j

    rng = np.random.RandomState(314159)
    starts = [base]

    # The triangular lattice is an excellent incumbent, but its active contact
    # graph makes small perturbations return to the same 1/13 local packing.
    # Include both substantial lattice deformations and genuinely unrelated
    # feasible (after normalization) point sets so SLSQP can reach packing
    # graphs with a slightly smaller diameter.
    for k in range(24):
        amp = 0.08 + 0.32 * k / 23.0
        starts.append(normalize(base + rng.normal(0.0, amp, base.shape)))

    # Add concentric-ring configurations.  These have much better initial
    # separation than arbitrary square clouds and expose non-lattice basins.
    for nr, na in ((1, 5), (1, 6), (2, 5), (2, 6), (3, 4)):
        x = [[0.0, 0.0]]
        for k in range(nr):
            radius = (k + 1.0) / (nr + 1.0)
            for j in range(na if k < nr - 1 else 16 - len(x)):
                a = 2.0 * np.pi * (j + 0.5 * (k & 1)) / na
                x.append([radius * np.cos(a), radius * np.sin(a)])
        starts.append(normalize(np.asarray(x[:16], float)))

    # Greedy farthest-point starts are inexpensive and are substantially more
    # useful than clouds containing accidental near-collisions.
    for _ in range(36):
        cand = rng.uniform(-1.0, 1.0, (1200, 2))
        chosen = [int(rng.randint(len(cand)))]
        md = np.full(len(cand), np.inf)
        for _j in range(15):
            delta = cand - cand[chosen[-1]]
            md = np.minimum(md, (delta * delta).sum(1))
            md[chosen] = -1.0
            chosen.append(int(np.argmax(md)))
        starts.append(normalize(cand[chosen]))

    for x in starts:
        z = np.r_[x.ravel(), ratio(x)]
        res = minimize(fun, z, jac=jacfun, method="SLSQP",
                       constraints={"type": "ineq", "fun": con, "jac": jaccon},
                       options={"maxiter": 1200, "ftol": 3e-12, "disp": False})
        y = normalize(res.x[:-1].reshape(16, 2))
        val = ratio(y)
        if val > bestv:
            best, bestv = y, val
    return best