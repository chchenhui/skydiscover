import numpy as np


def circle_packing21() -> np.ndarray:
    from itertools import permutations
    from scipy.optimize import minimize

    n = 21
    ii, jj = np.tril_indices(n, -1)
    m = len(ii)

    def objective(v):
        return -float(np.sum(v[2:3 * n:3]))

    def objective_jac(v):
        g = np.zeros_like(v)
        g[2:3 * n:3] = -1.0
        return g

    def con_square(z):
        q = z.reshape(n, 3)
        dx = q[ii, 0] - q[jj, 0]
        dy = q[ii, 1] - q[jj, 1]
        return np.r_[
            q[:, 0] - q[:, 2],
            1.0 - q[:, 0] - q[:, 2],
            q[:, 1] - q[:, 2],
            1.0 - q[:, 1] - q[:, 2],
            np.hypot(dx, dy) - q[ii, 2] - q[jj, 2],
        ]

    def jac_square(z):
        q = z.reshape(n, 3)
        a = np.zeros((4 * n + m, 3 * n))
        k = np.arange(n)
        a[k, 3 * k] = 1.0
        a[k, 3 * k + 2] = -1.0
        a[n + k, 3 * k] = -1.0
        a[n + k, 3 * k + 2] = -1.0
        a[2 * n + k, 3 * k + 1] = 1.0
        a[2 * n + k, 3 * k + 2] = -1.0
        a[3 * n + k, 3 * k + 1] = -1.0
        a[3 * n + k, 3 * k + 2] = -1.0
        dx = q[ii, 0] - q[jj, 0]
        dy = q[ii, 1] - q[jj, 1]
        d = np.maximum(np.hypot(dx, dy), 1e-14)
        rows = 4 * n + np.arange(m)
        a[rows, 3 * ii] = dx / d
        a[rows, 3 * jj] = -dx / d
        a[rows, 3 * ii + 1] = dy / d
        a[rows, 3 * jj + 1] = -dy / d
        a[rows, 3 * ii + 2] = -1.0
        a[rows, 3 * jj + 2] = -1.0
        return a

    def con_aspect(v):
        q = v[:-1].reshape(n, 3)
        w = float(v[-1])
        h = 2.0 - w
        dx = q[ii, 0] - q[jj, 0]
        dy = q[ii, 1] - q[jj, 1]
        return np.r_[
            q[:, 0] - q[:, 2],
            w - q[:, 0] - q[:, 2],
            q[:, 1] - q[:, 2],
            h - q[:, 1] - q[:, 2],
            np.hypot(dx, dy) - q[ii, 2] - q[jj, 2],
        ]

    def jac_aspect(v):
        q = v[:-1].reshape(n, 3)
        a = np.zeros((4 * n + m, 3 * n + 1))
        k = np.arange(n)
        a[k, 3 * k] = 1.0
        a[k, 3 * k + 2] = -1.0
        a[n + k, 3 * k] = -1.0
        a[n + k, 3 * k + 2] = -1.0
        a[n + k, -1] = 1.0
        a[2 * n + k, 3 * k + 1] = 1.0
        a[2 * n + k, 3 * k + 2] = -1.0
        a[3 * n + k, 3 * k + 1] = -1.0
        a[3 * n + k, 3 * k + 2] = -1.0
        a[3 * n + k, -1] = -1.0
        dx = q[ii, 0] - q[jj, 0]
        dy = q[ii, 1] - q[jj, 1]
        d = np.maximum(np.hypot(dx, dy), 1e-14)
        rows = 4 * n + np.arange(m)
        a[rows, 3 * ii] = dx / d
        a[rows, 3 * jj] = -dx / d
        a[rows, 3 * ii + 1] = dy / d
        a[rows, 3 * jj + 1] = -dy / d
        a[rows, 3 * ii + 2] = -1.0
        a[rows, 3 * jj + 2] = -1.0
        return a

    def certified_scale(q, w, h):
        """Inflate fixed-center radii by the exact feasible uniform slack factor."""
        q = np.asarray(q, dtype=float).reshape(n, 3).copy()
        if not np.all(np.isfinite(q)) or w <= 0.0 or h <= 0.0:
            return None
        q[:, 2] = np.maximum(q[:, 2], 1e-12)
        d = np.hypot(q[ii, 0] - q[jj, 0], q[ii, 1] - q[jj, 1])
        ratios = np.r_[
            q[:, 0] / q[:, 2],
            (w - q[:, 0]) / q[:, 2],
            q[:, 1] / q[:, 2],
            (h - q[:, 1]) / q[:, 2],
            d / (q[ii, 2] + q[jj, 2]),
        ]
        t = float(np.min(ratios))
        if not np.isfinite(t) or t <= 0.0:
            return None
        # The minimum ratio is the exact one-variable uniform inflation
        # optimum for these fixed centers.  Do not cap t at one: SLSQP
        # candidates can retain genuine feasible slack.
        q[:, 2] *= t * (1.0 - 1e-9)
        return q

    fallback = np.array([
        (x, y, 0.02)
        for y in (1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0)
        for x in np.linspace(1.0 / 14.0, 13.0 / 14.0, 7)
    ], dtype=float)

    best_q = fallback
    best_w = 1.0
    best_val = float(np.sum(fallback[:, 2]))

    base = {
        (6, 5, 5, 5), (5, 6, 5, 5), (5, 5, 6, 5), (5, 5, 5, 6),
        (5, 4, 4, 4, 4), (4, 5, 4, 4, 4), (4, 4, 5, 4, 4),
        (4, 4, 4, 5, 4), (4, 4, 4, 4, 5),
    }
    base.update(set(permutations((4, 4, 4, 3, 3, 3))))
    patterns = sorted(base, key=lambda z: (len(z), z))

    square_bounds = [(0.0, 1.0), (0.0, 1.0), (1e-10, 0.5)] * n
    square_cons = {"type": "ineq", "fun": con_square, "jac": jac_square}
    square_seeds = []

    for rows in patterns:
        for phase in (0.0, 0.018, 0.041):
            pts = []
            nr = len(rows)
            for row, count in enumerate(rows):
                x = np.linspace(1.0 / (2 * count), 1.0 - 1.0 / (2 * count), count)
                x = np.clip(x + (phase if row & 1 else -phase), 0.015, 0.985)
                pts.extend((xx, (row + 0.5) / nr) for xx in x)
            z0 = np.array([(x, y, 0.052) for x, y in pts]).ravel()
            try:
                res = minimize(
                    objective, z0, jac=objective_jac, method="SLSQP",
                    bounds=square_bounds, constraints=square_cons,
                    options={"maxiter": 1300, "ftol": 2e-12, "disp": False},
                )
                raw = res.x if np.all(np.isfinite(res.x)) else z0
            except Exception:
                raw = z0
            q = certified_scale(raw, 1.0, 1.0)
            if q is not None:
                square_seeds.append(q)
                value = float(np.sum(q[:, 2]))
                if value > best_val:
                    best_q, best_w, best_val = q, 1.0, value

    square_seeds.sort(key=lambda q: float(np.sum(q[:, 2])), reverse=True)
    square_seeds = square_seeds[:12] if square_seeds else [best_q.copy()]

    aspect_bounds = [(0.0, 1.5), (0.0, 1.5), (1e-10, 0.5)] * n + [(0.55, 1.45)]
    aspect_cons = {"type": "ineq", "fun": con_aspect, "jac": jac_aspect}
    pool = []

    for q0 in square_seeds:
        for w0 in (0.78, 0.86, 0.94, 1.0, 1.06, 1.14, 1.22):
            h0 = 2.0 - w0
            start = q0.copy()
            start[:, 0] *= w0
            start[:, 1] *= h0
            start[:, 2] *= min(w0, h0) * (1.0 - 1e-9)
            try:
                res = minimize(
                    objective, np.r_[start.ravel(), w0], jac=objective_jac,
                    method="SLSQP", bounds=aspect_bounds, constraints=aspect_cons,
                    options={"maxiter": 1900, "ftol": 8e-13, "disp": False},
                )
                if not np.all(np.isfinite(res.x)):
                    continue
                w = float(res.x[-1])
                q = certified_scale(res.x[:-1], w, 2.0 - w)
                if q is None:
                    continue
                pool.append((q, w))
                value = float(np.sum(q[:, 2]))
                if value > best_val:
                    best_q, best_w, best_val = q, w, value
            except Exception:
                pass

    pool.append((best_q.copy(), best_w))
    pool.sort(key=lambda qw: float(np.sum(qw[0][:, 2])), reverse=True)

    for q0, w0 in pool[:6]:
        try:
            res = minimize(
                objective, np.r_[q0.ravel(), w0], jac=objective_jac,
                method="SLSQP", bounds=aspect_bounds, constraints=aspect_cons,
                options={"maxiter": 2600, "ftol": 3e-13, "disp": False},
            )
            if not np.all(np.isfinite(res.x)):
                continue
            w = float(res.x[-1])
            q = certified_scale(res.x[:-1], w, 2.0 - w)
            if q is not None and float(np.sum(q[:, 2])) > best_val:
                best_q, best_w, best_val = q, w, float(np.sum(q[:, 2]))
        except Exception:
            pass

    # Active-contact cavity subsystem replacement.  The incumbent is retained
    # unless a repaired cavity is globally feasible and strictly better.
    incumbent = certified_scale(best_q, best_w, 2.0 - best_w)
    if incumbent is None:
        incumbent = best_q.copy()
    incumbent = np.asarray(incumbent, dtype=float)
    incumbent_sum = float(np.sum(incumbent[:, 2]))
    wfix = float(best_w)
    hfix = 2.0 - wfix

    # Build the contact graph from both wall and circle clearances.
    wall_slack = np.r_[
        incumbent[:, 0] - incumbent[:, 2],
        wfix - incumbent[:, 0] - incumbent[:, 2],
        incumbent[:, 1] - incumbent[:, 2],
        hfix - incumbent[:, 1] - incumbent[:, 2],
    ].reshape(4, n).T
    pair_slack = np.hypot(
        incumbent[ii, 0] - incumbent[jj, 0],
        incumbent[ii, 1] - incumbent[jj, 1],
    ) - incumbent[ii, 2] - incumbent[jj, 2]

    graph = [set() for _ in range(n)]
    active_pairs = np.flatnonzero(pair_slack < 2.0e-5)
    for p in active_pairs:
        a = int(ii[p])
        b = int(jj[p])
        graph[a].add(b)
        graph[b].add(a)

    # Deterministic cavity candidates: seed plus graph-distance-one
    # neighbors, restricted to the requested mesoscopic size.
    cavities = []
    for seed in range(n):
        members = sorted({seed}.union(graph[seed]))
        if 4 <= len(members) <= 8:
            cavities.append(tuple(members))
    cavities = sorted(set(cavities), key=lambda z: (len(z), z))

    for members in cavities:
        movable = np.asarray(members, dtype=int)
        frozen = np.asarray(
            [k for k in range(n) if k not in set(members)], dtype=int
        )
        lm = len(movable)
        local_i, local_j = np.tril_indices(lm, -1)

        def cavity_objective(v):
            return -float(np.sum(v[2:3 * lm:3]))

        def cavity_objective_jac(v):
            g = np.zeros(3 * lm, dtype=float)
            g[2:3 * lm:3] = -1.0
            return g

        def cavity_constraints(v):
            z = v.reshape(lm, 3)
            out = [
                z[:, 0] - z[:, 2],
                wfix - z[:, 0] - z[:, 2],
                z[:, 1] - z[:, 2],
                hfix - z[:, 1] - z[:, 2],
            ]
            if len(local_i):
                dx = z[local_i, 0] - z[local_j, 0]
                dy = z[local_i, 1] - z[local_j, 1]
                out.append(np.hypot(dx, dy) -
                           z[local_i, 2] - z[local_j, 2])
            if len(frozen):
                dx = z[:, None, 0] - incumbent[frozen, 0][None, :]
                dy = z[:, None, 1] - incumbent[frozen, 1][None, :]
                out.append((
                    np.hypot(dx, dy) -
                    z[:, None, 2] -
                    incumbent[frozen, 2][None, :]
                ).ravel())
            return np.concatenate(out)

        def cavity_constraints_jac(v):
            z = v.reshape(lm, 3)
            nf = len(frozen)
            rows = 4 * lm + len(local_i) + lm * nf
            J = np.zeros((rows, 3 * lm), dtype=float)
            k = np.arange(lm)

            J[k, 3 * k] = 1.0
            J[k, 3 * k + 2] = -1.0
            J[lm + k, 3 * k] = -1.0
            J[lm + k, 3 * k + 2] = -1.0
            J[2 * lm + k, 3 * k + 1] = 1.0
            J[2 * lm + k, 3 * k + 2] = -1.0
            J[3 * lm + k, 3 * k + 1] = -1.0
            J[3 * lm + k, 3 * k + 2] = -1.0

            row = 4 * lm
            if len(local_i):
                dx = z[local_i, 0] - z[local_j, 0]
                dy = z[local_i, 1] - z[local_j, 1]
                d = np.maximum(np.hypot(dx, dy), 1.0e-14)
                J[row + np.arange(len(local_i)), 3 * local_i] = dx / d
                J[row + np.arange(len(local_i)), 3 * local_j] = -dx / d
                J[row + np.arange(len(local_i)), 3 * local_i + 1] = dy / d
                J[row + np.arange(len(local_i)), 3 * local_j + 1] = -dy / d
                J[row + np.arange(len(local_i)), 3 * local_i + 2] = -1.0
                J[row + np.arange(len(local_i)), 3 * local_j + 2] = -1.0
                row += len(local_i)

            if nf:
                dx = z[:, None, 0] - incumbent[frozen, 0][None, :]
                dy = z[:, None, 1] - incumbent[frozen, 1][None, :]
                d = np.maximum(np.hypot(dx, dy), 1.0e-14).ravel()
                dx = dx.ravel()
                dy = dy.ravel()
                for a in range(lm):
                    sl = row + a * nf + np.arange(nf)
                    J[sl, 3 * a] = dx[a * nf:(a + 1) * nf] / d[
                        a * nf:(a + 1) * nf
                    ]
                    J[sl, 3 * a + 1] = dy[a * nf:(a + 1) * nf] / d[
                        a * nf:(a + 1) * nf
                    ]
                    J[sl, 3 * a + 2] = -1.0
            return J

        starts = [incumbent[movable].copy()]
        reflected = starts[0].copy()
        reflected[:, 0] = wfix - reflected[:, 0]
        reflected = reflected[::-1].copy()
        starts.append(reflected)

        for start in starts:
            try:
                sol = minimize(
                    cavity_objective,
                    start.ravel(),
                    jac=cavity_objective_jac,
                    method="SLSQP",
                    bounds=[(0.0, wfix), (0.0, hfix), (1.0e-10, 0.5)] * lm,
                    constraints={
                        "type": "ineq",
                        "fun": cavity_constraints,
                        "jac": cavity_constraints_jac,
                    },
                    options={"maxiter": 900, "ftol": 5.0e-13, "disp": False},
                )
            except Exception:
                continue
            if not np.all(np.isfinite(sol.x)):
                continue

            trial = incumbent.copy()
            trial[movable] = sol.x.reshape(lm, 3)
            repaired = certified_scale(trial, wfix, hfix)
            if repaired is None:
                continue

            dx = repaired[ii, 0] - repaired[jj, 0]
            dy = repaired[ii, 1] - repaired[jj, 1]
            globally_valid = (
                np.all(repaired[:, 2] > 0.0) and
                np.all(repaired[:, 0] >= repaired[:, 2] - 1.0e-10) and
                np.all(repaired[:, 0] + repaired[:, 2] <= wfix + 1.0e-10) and
                np.all(repaired[:, 1] >= repaired[:, 2] - 1.0e-10) and
                np.all(repaired[:, 1] + repaired[:, 2] <= hfix + 1.0e-10) and
                np.all(np.hypot(dx, dy) >=
                       repaired[ii, 2] + repaired[jj, 2] - 1.0e-10)
            )
            value = float(np.sum(repaired[:, 2]))
            if globally_valid and value > max(incumbent_sum, 2.36583230494):
                incumbent = repaired
                incumbent_sum = value

    # Repeat certified scaling on the final selected result.
    q = certified_scale(incumbent, wfix, hfix)
    return np.asarray(q if q is not None else incumbent, dtype=float)


if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:, 2]):.15f}")