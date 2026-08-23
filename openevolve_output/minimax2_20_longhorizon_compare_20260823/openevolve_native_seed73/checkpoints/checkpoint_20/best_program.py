# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Construct a diameter-one 16-point packing using deterministic SLSQP restarts with exact constraint Jacobians."""
    # A structured non-random feasible starting point:
    # one central point, five inner points, and ten outer points.
    #
    # The radii are chosen so that the central-to-inner and nearest
    # inner-to-outer distances are both one before normalization.
    inner_radius = 1.0
    outer_radius = 2.0 * np.cos(np.pi / 10.0)

    inner_angles = np.pi / 10.0 + 2.0 * np.pi * np.arange(5) / 5.0
    outer_angles = 2.0 * np.pi * np.arange(10) / 10.0

    points = np.vstack(
        [
            np.array([[0.0, 0.0]]),
            inner_radius
            * np.column_stack((np.cos(inner_angles), np.sin(inner_angles))),
            outer_radius
            * np.column_stack((np.cos(outer_angles), np.sin(outer_angles))),
        ]
    )

    pairs = np.array(
        [(i, j) for i in range(16) for j in range(i + 1, 16)],
        dtype=int,
    )

    def pair_distances(p):
        delta = p[pairs[:, 0]] - p[pairs[:, 1]]
        return np.sqrt(np.sum(delta * delta, axis=1))

    # Normalize the initial configuration to maximum pair distance one.
    points /= np.max(pair_distances(points))
    best = points.copy()
    best_ratio = np.min(pair_distances(best)) / np.max(pair_distances(best))

    # SLSQP is used only as a deterministic local improvement.  Keeping a
    # valid analytic construction as fallback makes the result robust if
    # SciPy is unavailable or a numerical solve does not improve it.
    try:
        from scipy.optimize import minimize

        # Point zero remains at the origin, removing translation freedom.
        # The final scalar is the common separation lower bound.
        def unpack(x):
            return np.vstack((np.zeros((1, 2)), x[:-1].reshape(15, 2)))

        def objective(x):
            return -x[-1]

        def constraints(x):
            p = unpack(x)
            delta = p[pairs[:, 0]] - p[pairs[:, 1]]
            dist2 = np.sum(delta * delta, axis=1)
            separation = x[-1]
            # Simultaneously enforce separation <= distance <= diameter.
            return np.concatenate((dist2 - separation * separation, 1.0 - dist2))

        def constraint_jacobian(x):
            """Exact derivatives of squared lower- and upper-distance bounds."""
            p = unpack(x)
            delta = p[pairs[:, 0]] - p[pairs[:, 1]]
            separation = x[-1]
            m = len(pairs)
            jac = np.zeros((2 * m, 31), dtype=float)

            # For ||p_i-p_j||^2, derivatives are 2*(p_i-p_j) and its
            # negative.  Point zero is fixed, so it has no optimization slot.
            for row, (i, j) in enumerate(pairs):
                derivative = 2.0 * delta[row]
                if i:
                    jac[row, 2 * (i - 1) : 2 * i] = derivative
                if j:
                    jac[row, 2 * (j - 1) : 2 * j] = -derivative
                jac[row, -1] = -2.0 * separation
                jac[m + row, :-1] = -jac[row, :-1]

            return jac

        # The symmetric seed is useful, but it can lock SLSQP into a weaker
        # symmetric contact graph.  These reproducible perturbations expose
        # nearby asymmetric graphs without introducing nondeterminism.
        rng = np.random.default_rng(16031991)
        starts = [points.copy()]

        # Perturbations around the strong 1+5+10 construction search nearby
        # asymmetric active-contact graphs.  The number of small restarts is
        # deliberately larger because those starts are substantially more
        # likely to remain in the basin of a good diameter packing.
        for amplitude, count in (
            (0.002, 8),
            (0.005, 10),
            (0.012, 12),
            (0.028, 12),
            (0.060, 12),
            (0.110, 12),
            (0.180, 10),
            (0.280, 8),
        ):
            for _ in range(count):
                trial = points + amplitude * rng.standard_normal(points.shape)
                trial -= trial[0]
                trial /= np.max(pair_distances(trial))
                starts.append(trial)

        # Uniform disk samples are preferable to Gaussian clouds here: a
        # diameter-constrained optimum uses the available boundary area, while
        # Gaussian samples overpopulate the centre and often converge to weak
        # local contact graphs.
        for _ in range(48):
            angles = 2.0 * np.pi * rng.random(16)
            radii = np.sqrt(rng.random(16))
            trial = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
            trial -= trial[0]
            trial /= np.max(pair_distances(trial))
            starts.append(trial)

        # Add jittered triangular-lattice patches.  Hexagonal local structure
        # is common in extremal planar packings and provides a qualitatively
        # different basin from both circular-ring and random-disk starts.
        lattice = np.array(
            [
                (i + 0.5 * (j & 1), np.sqrt(3.0) * 0.5 * j)
                for j in range(-2, 3)
                for i in range(-2, 3)
            ],
            dtype=float,
        )
        for _ in range(32):
            chosen = lattice[rng.choice(len(lattice), size=16, replace=False)]
            trial = chosen + 0.16 * rng.standard_normal((16, 2))
            trial -= trial[0]
            trial /= np.max(pair_distances(trial))
            starts.append(trial)

        for trial in starts:
            x0 = np.concatenate(
                (trial[1:].ravel(), [np.min(pair_distances(trial))])
            )
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                constraints={
                    "type": "ineq",
                    "fun": constraints,
                    "jac": constraint_jacobian,
                },
                options={"maxiter": 1400, "ftol": 2e-13, "disp": False},
            )

            if not np.all(np.isfinite(result.x)):
                continue

            candidate = unpack(result.x)
            distances = pair_distances(candidate)
            candidate_ratio = np.min(distances) / np.max(distances)

            # Ratios are scale invariant; the final normalization below
            # removes negligible diameter constraint tolerance exactly.
            if candidate_ratio > best_ratio:
                best_ratio = candidate_ratio
                best = candidate
    except Exception:
        pass

    # Final normalization removes tiny optimizer constraint tolerances.
    best /= np.max(pair_distances(best))
    return best


# EVOLVE-BLOCK-END
