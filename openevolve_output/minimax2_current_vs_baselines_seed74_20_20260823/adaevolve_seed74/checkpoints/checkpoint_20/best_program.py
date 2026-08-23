# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Construct a dense 16-point packing, then solve its active contacts exactly.

    COBYLA first identifies a high-quality alternating-corner triangular packing.
    A deterministic least-squares active-set polish then pins a diameter pair,
    equalizes all identified nearest-neighbor squared distances, and restores
    every identified diameter to unit length before full pairwise validation.
    """
    removed = {(2, 0), (-2, 2), (0, -2)}
    axial = [
        (q, r)
        for q in range(-2, 3)
        for r in range(-2, 3)
        if max(abs(q), abs(r), abs(q + r)) <= 2 and (q, r) not in removed
    ]
    lattice = np.array(
        [(q + 0.5 * r, 0.5 * np.sqrt(3.0) * r) for q, r in axial],
        dtype=np.float64,
    )
    lattice -= lattice[0]
    ij = np.triu_indices(16, 1)

    def normalize(p):
        delta = p[:, None, :] - p[None, :, :]
        return p / np.sqrt(np.max(np.sum(delta * delta, axis=-1)))

    def value(p):
        delta = p[:, None, :] - p[None, :, :]
        d2 = np.sum(delta * delta, axis=-1)[ij]
        return float(np.min(d2) / np.max(d2))

    best = normalize(lattice)
    best_value = value(best)

    try:
        from scipy.optimize import minimize

        def unpack(v):
            return np.vstack((np.zeros(2), v[:-1].reshape(15, 2)))

        def constraints(v):
            p = unpack(v)
            delta = p[:, None, :] - p[None, :, :]
            d2 = np.sum(delta * delta, axis=-1)[ij]
            return np.concatenate((d2 - v[-1] * v[-1], 1.0 - d2))

        rng = np.random.default_rng(314159265)
        starts = [best]
        starts.extend(
            normalize(
                np.vstack((
                    np.zeros(2),
                    best[1:] + rng.normal(0.0, scale, size=(15, 2)),
                ))
            )
            for scale in (0.025, 0.050, 0.085)
            for _ in range(3)
        )

        for start in starts:
            # Starting below the actual nearest spacing makes all lower-bound
            # constraints feasible, which is substantially more reliable for
            # COBYLA than beginning on the lattice's many active contacts.
            result = minimize(
                lambda v: -v[-1],
                np.r_[start[1:].ravel(), 0.18],
                method="COBYLA",
                constraints={"type": "ineq", "fun": constraints},
                options={
                    "maxiter": 18000,
                    "rhobeg": 0.04,
                    "tol": 2e-10,
                    "catol": 2e-9,
                },
            )
            candidate = normalize(unpack(result.x))
            candidate_value = value(candidate)
            if np.isfinite(candidate_value) and candidate_value > best_value:
                best, best_value = candidate, candidate_value
    except Exception:
        pass

    # COBYLA is effective at discovering the contact topology, but its final
    # coordinates can retain small inequality-convergence errors.  Solve the
    # inferred jammed active graph directly using squared-distance equations.
    # This is deliberately performed only after the exploratory optimization:
    # a least-squares solve is a polisher, not a replacement for topology search.
    try:
        from scipy.optimize import least_squares

        delta = best[:, None, :] - best[None, :, :]
        all_d2 = np.sum(delta * delta, axis=-1)
        current_dmax2 = float(np.max(all_d2))
        current_dmin2 = float(np.min(all_d2[ij]))

        # Select a diameter pair for translation, rotation, and scale gauges.
        diameter_index = int(np.argmax(all_d2[iu := np.triu_indices(16, 1)]))
        anchor_a = int(iu[0][diameter_index])
        anchor_b = int(iu[1][diameter_index])

        # Put the selected diameter at (0,0)--(1,0).  This removes all three
        # similarity freedoms without adding artificial residual equations.
        diameter_vector = best[anchor_b] - best[anchor_a]
        diameter_length = float(np.linalg.norm(diameter_vector))
        c = diameter_vector[0] / diameter_length
        s = diameter_vector[1] / diameter_length
        shifted = (best - best[anchor_a]) / diameter_length
        gauged = np.empty_like(shifted)
        gauged[:, 0] = c * shifted[:, 0] + s * shifted[:, 1]
        gauged[:, 1] = -s * shifted[:, 0] + c * shifted[:, 1]
        gauged[anchor_a] = (0.0, 0.0)
        gauged[anchor_b] = (1.0, 0.0)

        gd = gauged[:, None, :] - gauged[None, :, :]
        gd2 = np.sum(gd * gd, axis=-1)[ij]
        gmin = float(np.min(gd2))
        gmax = float(np.max(gd2))

        # Include contacts conservatively.  The threshold is much larger than
        # COBYLA's requested feasibility tolerance, while remaining far below
        # the gap to the next triangular-lattice distance level.
        active_min = gd2 <= gmin + 5.0e-6
        active_max = gd2 >= gmax - 5.0e-6
        min_pairs = np.column_stack((ij[0][active_min], ij[1][active_min]))
        max_pairs = np.column_stack((ij[0][active_max], ij[1][active_max]))

        # A jammed graph should have enough equations to constrain the 28 free
        # coordinates plus the common nearest-contact squared distance.
        free_indices = np.array(
            [k for k in range(16) if k != anchor_a and k != anchor_b],
            dtype=np.int64,
        )
        if len(min_pairs) + len(max_pairs) >= 29:
            def unpack_active(x):
                p = np.empty((16, 2), dtype=np.float64)
                p[anchor_a] = (0.0, 0.0)
                p[anchor_b] = (1.0, 0.0)
                p[free_indices] = x[:-1].reshape(14, 2)
                return p

            def active_residual(x):
                p = unpack_active(x)
                q = p[:, None, :] - p[None, :, :]
                q2 = np.sum(q * q, axis=-1)
                t = x[-1]
                return np.concatenate((
                    q2[min_pairs[:, 0], min_pairs[:, 1]] - t,
                    q2[max_pairs[:, 0], max_pairs[:, 1]] - 1.0,
                ))

            x0 = np.r_[gauged[free_indices].ravel(), gmin]
            polished_result = least_squares(
                active_residual,
                x0,
                method="trf",
                x_scale="jac",
                ftol=1.0e-14,
                xtol=1.0e-14,
                gtol=1.0e-14,
                max_nfev=30000,
            )
            polished = unpack_active(polished_result.x)

            # Normalize once more before evaluating all 120 distances.  Accept
            # only a genuinely feasible active-set solution; in particular,
            # omitted contacts are not allowed to become shorter or longer.
            pd = polished[:, None, :] - polished[None, :, :]
            pd2_full = np.sum(pd * pd, axis=-1)
            scale = np.sqrt(float(np.max(pd2_full)))
            polished /= scale
            pd = polished[:, None, :] - polished[None, :, :]
            pd2_full = np.sum(pd * pd, axis=-1)
            pd2 = pd2_full[ij]
            solved_t = float(polished_result.x[-1] / (scale * scale))

            inactive_min_ok = np.all(
                pd2[~active_min] >= solved_t - 2.0e-9
            )
            inactive_max_ok = np.all(
                pd2[~active_max] <= 1.0 + 2.0e-9
            )
            residual_ok = (
                np.max(np.abs(active_residual(polished_result.x))) < 2.0e-7
            )
            polished_value = float(np.min(pd2) / np.max(pd2))

            if (
                residual_ok
                and inactive_min_ok
                and inactive_max_ok
                and np.isfinite(polished_value)
                and polished_value > best_value
            ):
                best = polished
                best_value = polished_value
    except Exception:
        # The lattice/COBYLA candidate remains a valid deterministic fallback
        # if SciPy lacks least_squares or an ill-conditioned active graph occurs.
        pass

    return np.asarray(best, dtype=np.float64)


# EVOLVE-BLOCK-END
