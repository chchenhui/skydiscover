# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """Use deterministic multistart SQP on 21 variable circles in a variable-perimeter box."""
    from scipy.optimize import minimize

    n = 21
    m = n * (n - 1) // 2
    rng = np.random.default_rng(20260823)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def con(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:3 * n]
        w = z[-1]
        h = 2.0 - w
        out = np.empty(4 * n + 2 + m)
        out[:n] = x - r
        out[n:2 * n] = w - x - r
        out[2 * n:3 * n] = y - r
        out[3 * n:4 * n] = h - y - r
        out[4 * n] = w - 1e-5
        out[4 * n + 1] = 1.99999 - w
        k = 4 * n + 2
        for i, j in pairs:
            out[k] = ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                      - (r[i] + r[j]) ** 2)
            k += 1
        return out

    def con_jac(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:3 * n]
        jac = np.zeros((4 * n + 2 + m, 3 * n + 1))
        ii = np.arange(n)
        jac[ii, ii] = 1.0
        jac[ii, 2 * n + ii] = -1.0
        jac[n + ii, ii] = -1.0
        jac[n + ii, 2 * n + ii] = -1.0
        jac[n + ii, -1] = 1.0
        jac[2 * n + ii, n + ii] = 1.0
        jac[2 * n + ii, 2 * n + ii] = -1.0
        jac[3 * n + ii, n + ii] = -1.0
        jac[3 * n + ii, 2 * n + ii] = -1.0
        jac[3 * n + ii, -1] = -1.0
        jac[4 * n, -1] = 1.0
        jac[4 * n + 1, -1] = -1.0
        k = 4 * n + 2
        for i, j in pairs:
            dx, dy, sr = x[i] - x[j], y[i] - y[j], r[i] + r[j]
            jac[k, i], jac[k, j] = 2.0 * dx, -2.0 * dx
            jac[k, n + i], jac[k, n + j] = 2.0 * dy, -2.0 * dy
            jac[k, 2 * n + i] = jac[k, 2 * n + j] = -2.0 * sr
            k += 1
        return jac

    def make_start(u, w, variation):
        """Turn normalized centers into a strictly feasible, varied initial packing."""
        h = 2.0 - w
        x = np.clip(u[:, 0] * w, 0.015, w - 0.015)
        y = np.clip(u[:, 1] * h, 0.015, h - 0.015)
        q = np.minimum.reduce((x, w - x, y, h - y))
        for i, j in pairs:
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            q[i] = min(q[i], 0.5 * d)
            q[j] = min(q[j], 0.5 * d)
        r = q * (0.82 + variation * rng.uniform(-1.0, 1.0, n))
        r = np.maximum(r, 0.004)
        z = np.r_[x, y, r, w]
        # One scalar repair makes every deliberately perturbed seed feasible.
        s = min(1.0, np.min(con(z)[:4 * n + 2 + m] /
                            np.maximum(np.r_[r, r, r, r, np.ones(2),
                                             np.ones(m) * 0.02], 1e-12)))
        if s < 1.0:
            z[2 * n:3 * n] *= max(0.05, 0.92 * s)
        return z

    # A 5 by 5 pattern with the four corners removed is substantially better
    # than the former three-row topology.  Staggered versions let SQP discover
    # asymmetric large boundary circles instead of preserving that topology.
    grid = np.array([(i / 4.0, j / 4.0)
                     for j in range(5) for i in range(5)
                     if (i, j) not in ((0, 0), (4, 0), (0, 4), (4, 4))],
                    dtype=float)
    hex_rows = []
    for row, count in enumerate((4, 5, 4, 5, 3)):
        xs = np.linspace(0.12, 0.88, count)
        if count == 4:
            xs += 0.035 * (1 if row % 2 else -1)
        hex_rows.extend((x, 0.10 + 0.20 * row) for x in xs)
    hexagon = np.asarray(hex_rows, dtype=float)

    starts = []
    for base in (grid, hexagon):
        for w in (0.90, 0.96, 1.00, 1.04, 1.10):
            starts.append(make_start(base, w, 0.0))
            for _ in range(2):
                noisy = np.clip(base + rng.normal(0.0, 0.045, base.shape),
                                0.025, 0.975)
                starts.append(make_start(noisy, w, 0.20))

    best = None
    best_score = -np.inf
    objective_jac = np.r_[np.zeros(2 * n), -np.ones(n), 0.0]
    for z0 in starts:
        result = minimize(
            lambda z: -np.sum(z[2 * n:3 * n]),
            z0, jac=lambda z: objective_jac, method="SLSQP",
            bounds=([(0.0, 2.0)] * (2 * n) +
                    [(1e-6, 0.5)] * n + [(1e-5, 1.99999)]),
            constraints={"type": "ineq", "fun": con, "jac": con_jac},
            options={"maxiter": 1100, "ftol": 3e-12, "disp": False},
        )
        if np.all(np.isfinite(result.x)):
            z = result.x
            if np.min(con(z)) >= -2e-7 and np.sum(z[2 * n:3 * n]) > best_score:
                best, best_score = z.copy(), np.sum(z[2 * n:3 * n])

    if best is None:
        best = starts[0].copy()

    x, y, r = best[:n].copy(), best[n:2 * n].copy(), best[2 * n:3 * n].copy()
    w = float(best[-1])
    h = 2.0 - w
    scale = min(np.min(x / r), np.min((w - x) / r),
                np.min(y / r), np.min((h - y) / r), 1.0)
    for i, j in pairs:
        scale = min(scale, np.hypot(x[i] - x[j], y[i] - y[j]) / (r[i] + r[j]))
    r *= max(0.0, scale) * (1.0 - 3e-8)
    return np.column_stack((x, y, r))


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
