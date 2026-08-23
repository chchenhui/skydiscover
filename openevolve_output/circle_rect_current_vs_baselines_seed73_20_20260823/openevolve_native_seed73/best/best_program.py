# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """Optimize 21 circles using analytic SLSQP derivatives and deterministic restarts.

    The variables are all centers, all radii, and rectangle width; height is
    2-width.  Feasible perturbed lattice and incumbent seeds explore distinct
    contact graphs, while analytic constraint Jacobians permit many restarts.
    """
    n = 21
    root3 = np.sqrt(3.0)
    ii, jj = np.triu_indices(n, 1)

    safe_r = (2.0 - 4.0e-7) / (10.0 + 5.0 * root3)
    base = []
    for row, count in enumerate((4, 3, 4, 3, 4, 3)):
        offset = safe_r if count == 4 else 2.0 * safe_r
        for col in range(count):
            base.append((
                offset + 2.0 * safe_r * col,
                safe_r + row * root3 * safe_r,
                safe_r,
            ))
    fallback = np.asarray(base, dtype=float)

    try:
        from scipy.optimize import minimize
    except Exception:
        return fallback

    def constraints(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:3 * n]
        width = z[-1]
        dx = x[ii] - x[jj]
        dy = y[ii] - y[jj]
        s = r[ii] + r[jj]
        return np.concatenate((
            x - r,
            width - x - r,
            y - r,
            2.0 - width - y - r,
            (dx * dx + dy * dy) / s - s,
        ))

    def constraints_jac(z):
        """Return the exact dense Jacobian of walls and pair clearances."""
        x, y, r = z[:n], z[n:2 * n], z[2 * n:3 * n]
        p = len(ii)
        jac = np.zeros((4 * n + p, 3 * n + 1), dtype=float)
        q = np.arange(n)

        jac[q, q] = 1.0
        jac[q, 2 * n + q] = -1.0

        jac[n + q, q] = -1.0
        jac[n + q, 2 * n + q] = -1.0
        jac[n + q, -1] = 1.0

        jac[2 * n + q, n + q] = 1.0
        jac[2 * n + q, 2 * n + q] = -1.0

        jac[3 * n + q, n + q] = -1.0
        jac[3 * n + q, 2 * n + q] = -1.0
        jac[3 * n + q, -1] = -1.0

        dx = x[ii] - x[jj]
        dy = y[ii] - y[jj]
        s = r[ii] + r[jj]
        d2 = dx * dx + dy * dy
        rows = 4 * n + np.arange(p)
        jac[rows, ii] = 2.0 * dx / s
        jac[rows, jj] = -2.0 * dx / s
        jac[rows, n + ii] = 2.0 * dy / s
        jac[rows, n + jj] = -2.0 * dy / s
        dr = -d2 / (s * s) - 1.0
        jac[rows, 2 * n + ii] = dr
        jac[rows, 2 * n + jj] = dr
        return jac

    bounds = (
        [(0.0, 2.0)] * (2 * n)
        + [(1.0e-5, 0.5)] * n
        + [(0.35, 1.65)]
    )
    objective = lambda z: -float(np.sum(z[2 * n:3 * n]))

    def objective_jac(z):
        """Return the exact gradient of negative total radius."""
        grad = np.zeros(3 * n + 1, dtype=float)
        grad[2 * n:3 * n] = -1.0
        return grad

    best = fallback
    best_sum = float(np.sum(fallback[:, 2]))
    best_z = None
    rng = np.random.default_rng(184729)
    widths = np.array((
        0.54, 0.59, 0.64, 0.69, 0.74, 0.79, 0.84, 0.89, 0.94,
        1.00, 1.06, 1.11, 1.16, 1.21, 1.26, 1.31, 1.36, 1.41,
        1.46,
    ))

    # The original 152 starts already finish quickly.  More deterministic
    # starts are substantially more valuable here than tightening SLSQP's
    # tolerance further, because distinct contact graphs give distinct optima.
    for attempt in range(2200):
        width = float(widths[attempt % len(widths)])
        height = 2.0 - width
        r0 = 0.90 * min(width / 8.0, height / (2.0 + 5.0 * root3))
        sx = 0.5 * (width - 8.0 * r0)
        sy = 0.5 * (height - (2.0 + 5.0 * root3) * r0)

        x0, y0 = [], []
        for row, count in enumerate((4, 3, 4, 3, 4, 3)):
            offset = r0 if count == 4 else 2.0 * r0
            for col in range(count):
                x0.append(sx + offset + 2.0 * r0 * col)
                y0.append(sy + r0 + row * root3 * r0)
        x0 = np.asarray(x0)
        y0 = np.asarray(y0)

        if attempt < len(widths):
            radii0 = np.full(n, r0)
        else:
            phase = (attempt - len(widths)) / (2199 - len(widths))
            radius_jitter = 0.16 + 0.14 * phase
            position_jitter = 0.008 + 0.030 * phase
            radii0 = r0 * (1.0 + rng.uniform(-radius_jitter, radius_jitter, n))
            x0 += rng.uniform(-position_jitter, position_jitter, n)
            y0 += rng.uniform(-position_jitter, position_jitter, n)

            ratios = np.concatenate((
                x0 / radii0,
                (width - x0) / radii0,
                y0 / radii0,
                (height - y0) / radii0,
            ))
            dx = x0[ii] - x0[jj]
            dy = y0[ii] - y0[jj]
            ratios = np.concatenate((
                ratios,
                np.sqrt(dx * dx + dy * dy) / (radii0[ii] + radii0[jj]),
            ))
            radii0 *= max(0.15, min(0.985, 0.995 * float(np.min(ratios))))

        # Every third late restart starts near the current incumbent rather
        # than near the six-row lattice.  This is useful because SLSQP is very
        # effective at polishing a fixed contact graph, but random lattice
        # starts alone rarely revisit promising asymmetric graphs.
        if best_z is not None and attempt >= 60 and attempt % 3 == 0:
            width = float(best_z[-1])
            height = 2.0 - width
            amp = 0.0025 if attempt % 11 else 0.012
            x0 = best_z[:n] + rng.uniform(-amp, amp, n)
            y0 = best_z[n:2 * n] + rng.uniform(-amp, amp, n)
            radii0 = best_z[2 * n:3 * n] * (
                1.0 + rng.uniform(-0.075, 0.075, n)
            )

            # Uniformly shrink the perturbed radii to make this seed feasible.
            # This avoids feeding SLSQP heavily overlapping configurations,
            # which otherwise tends to produce poor local stationary points.
            ratios = np.concatenate((
                x0 / radii0,
                (width - x0) / radii0,
                y0 / radii0,
                (height - y0) / radii0,
            ))
            dx = x0[ii] - x0[jj]
            dy = y0[ii] - y0[jj]
            ratios = np.concatenate((
                ratios,
                np.sqrt(dx * dx + dy * dy) / (radii0[ii] + radii0[jj]),
            ))
            radii0 *= max(0.12, min(0.992, 0.996 * float(np.min(ratios))))

        result = minimize(
            objective,
            np.r_[x0, y0, radii0, width],
            method="SLSQP",
            jac=objective_jac,
            bounds=bounds,
            constraints={
                "type": "ineq",
                "fun": constraints,
                "jac": constraints_jac,
            },
            options={"maxiter": 2600, "ftol": 2.0e-12, "disp": False},
        )

        z = result.x
        if not np.all(np.isfinite(z)):
            continue

        clearance = float(np.min(constraints(z)))
        radii = z[2 * n:3 * n].copy()
        if clearance < -1.0e-7 or np.min(radii) <= 2.0e-6:
            continue

        radii -= max(4.0e-7, -clearance + 4.0e-7)
        total = float(np.sum(radii))
        if np.min(radii) > 0.0 and total > best_sum:
            best = np.column_stack((z[:n], z[n:2 * n], radii))
            best_sum = total
            best_z = z.copy()

    return best


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
