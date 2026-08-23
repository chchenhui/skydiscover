# EVOLVE-BLOCK-START
import numpy as np


def circle_packing21() -> np.ndarray:
    """Optimize 21 variable-radius disks from many centered hexagonal seeds."""
    n = 21
    root3 = np.sqrt(3.0)
    ii, jj = np.triu_indices(n, 1)

    # Certified equal-radius packing, also used if SciPy is unavailable.
    safe_r = (2.0 - 4.0e-7) / (10.0 + 5.0 * root3)
    base = []
    for row, count in enumerate((4, 3, 4, 3, 4, 3)):
        offset = safe_r if count == 4 else 2.0 * safe_r
        for col in range(count):
            base.append((offset + 2.0 * safe_r * col,
                         safe_r + row * root3 * safe_r, safe_r))
    fallback = np.asarray(base, dtype=float)

    try:
        from scipy.optimize import minimize
    except Exception:
        return fallback

    def constraints(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:3 * n]
        w = z[-1]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        return np.concatenate((
            x - r, w - x - r, y - r, 2.0 - w - y - r,
            np.sqrt(dx[ii, jj] ** 2 + dy[ii, jj] ** 2) - r[ii] - r[jj],
        ))

    bounds = ([(0.0, 2.0)] * (2 * n) + [(1e-5, 0.5)] * n +
              [(0.35, 1.65)])
    objective = lambda z: -np.sum(z[2 * n:3 * n])

    best = fallback
    best_sum = float(np.sum(fallback[:, 2]))
    rng = np.random.default_rng(184729)

    # The prior starts occupied only the lower-left portion of the available
    # rectangle.  Centering a near-maximal hexagonal seed is materially better
    # conditioned: the optimizer begins with contacts near every side and can
    # then break symmetry through deterministic radius/position perturbations.
    widths = np.array((0.76, 0.82, 0.88, 0.94, 1.00, 1.08, 1.16, 1.24))

    for attempt in range(64):
        w0 = float(widths[attempt % len(widths)])
        h0 = 2.0 - w0
        # The 4,3,4,3,4,3 lattice has bounding dimensions 8r by
        # (2 + 5 sqrt(3))r.  Leave 10 percent room for asymmetric growth.
        r0 = 0.90 * min(w0 / 8.0, h0 / (2.0 + 5.0 * root3))
        sx = 0.5 * (w0 - 8.0 * r0)
        sy = 0.5 * (h0 - (2.0 + 5.0 * root3) * r0)

        x0, y0 = [], []
        for row, count in enumerate((4, 3, 4, 3, 4, 3)):
            offset = r0 if count == 4 else 2.0 * r0
            for col in range(count):
                x0.append(sx + offset + 2.0 * r0 * col)
                y0.append(sy + r0 + row * root3 * r0)
        x0 = np.asarray(x0)
        y0 = np.asarray(y0)

        if attempt < len(widths):
            # Include one exactly feasible symmetric start for each aspect ratio.
            rr = np.full(n, r0)
            xx, yy = x0, y0
        else:
            # Larger, aspect-ratio-diverse perturbations explore unequal-radius
            # contact graphs rather than repeatedly returning the same lattice.
            rr = r0 + rng.uniform(-0.009, 0.009, n)
            xx = x0 + rng.uniform(-0.007, 0.007, n)
            yy = y0 + rng.uniform(-0.007, 0.007, n)
        z0 = np.r_[xx, yy, rr, w0]

        result = minimize(
            objective, z0, method="SLSQP", bounds=bounds,
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 2200, "ftol": 3e-12, "disp": False},
        )
        z = result.x
        if not np.all(np.isfinite(z)):
            continue
        clearance = float(np.min(constraints(z)))
        r = z[2 * n:3 * n].copy()

        # Accept feasible iterates even if SLSQP reports a line-search status:
        # its final iterate is often a valid local packing in that situation.
        if clearance < -1e-7 or np.min(r) <= 2e-6:
            continue
        r -= max(4e-7, -clearance + 4e-7)
        total = float(np.sum(r))
        if np.min(r) > 0.0 and total > best_sum:
            best = np.column_stack((z[:n], z[n:2 * n], r))
            best_sum = total

    return best


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
