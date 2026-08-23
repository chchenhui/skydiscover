# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """Search deterministic, aspect-ratio-diverse feasible disk seeds with SLSQP.

    Pair separations are represented by a smooth squared-distance clearance,
    while increasingly perturbed hexagonal seeds explore unequal-radius contact
    graphs before a final uniform radius contraction certifies feasibility.
    """
    n = 21
    root3 = np.sqrt(3.0)
    ii, jj = np.triu_indices(n, 1)

    # Certified 4-3-4-3-4-3 equal-radius fallback.
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
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        return np.concatenate((
            x - r,
            width - x - r,
            y - r,
            2.0 - width - y - r,
            (dx[ii, jj] ** 2 + dy[ii, jj] ** 2
             - (r[ii] + r[jj]) ** 2) / (r[ii] + r[jj]),
        ))

    bounds = (
        [(0.0, 2.0)] * (2 * n)
        + [(1.0e-5, 0.5)] * n
        + [(0.35, 1.65)]
    )
    objective = lambda z: -float(np.sum(z[2 * n:3 * n]))

    best = fallback
    best_sum = float(np.sum(fallback[:, 2]))
    rng = np.random.default_rng(184729)
    widths = np.array((
        0.54, 0.59, 0.64, 0.69, 0.74, 0.79, 0.84, 0.89, 0.94,
        1.00, 1.06, 1.11, 1.16, 1.21, 1.26, 1.31, 1.36, 1.41,
        1.46,
    ))

    # The first pass retains well-conditioned near-lattice starts.  Subsequent
    # passes use progressively larger deterministic perturbations; this is
    # important because the best unequal-radius packings generally require
    # leaving the highly symmetric 4-3-4-3-4-3 contact basin.
    for attempt in range(152):
        width = float(widths[attempt % len(widths)])
        height = 2.0 - width

        # Center an interior lattice seed at each rectangle aspect ratio.
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
            # Perturb topology, then uniformly contract to make the start
            # genuinely feasible before calling the constrained solver.
            phase = (attempt - len(widths)) / max(1, 151 - len(widths))
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

        result = minimize(
            objective,
            np.r_[x0, y0, radii0, width],
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 2600, "ftol": 2.0e-12, "disp": False},
        )

        z = result.x
        if not np.all(np.isfinite(z)):
            continue

        clearance = float(np.min(constraints(z)))
        radii = z[2 * n:3 * n].copy()
        if clearance < -1.0e-7 or np.min(radii) <= 2.0e-6:
            continue

        # Uniform contraction preserves every constraint and gives robust
        # positive clearance against evaluator and solver roundoff.
        radii -= max(4.0e-7, -clearance + 4.0e-7)
        total = float(np.sum(radii))
        if np.min(radii) > 0.0 and total > best_sum:
            best = np.column_stack((z[:n], z[n:2 * n], radii))
            best_sum = total

    return best


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
