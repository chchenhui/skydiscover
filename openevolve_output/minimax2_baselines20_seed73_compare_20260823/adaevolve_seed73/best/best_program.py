# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, shgo


def min_max_dist_dim2_16() -> np.ndarray:
    """Use SHGO on an asymmetric 1+5+10 shell ansatz, then refine its best seeds."""
    inner_angles = 2.0 * np.pi * np.arange(5, dtype=np.float64) / 5.0
    outer_angles = np.pi / 10.0 + 2.0 * np.pi * np.arange(10, dtype=np.float64) / 10.0
    base = np.vstack((
        np.array([[0.0, 0.0]]),
        np.column_stack((np.cos(inner_angles), np.sin(inner_angles))),
        2.0 * np.cos(np.pi / 10.0) *
        np.column_stack((np.cos(outer_angles), np.sin(outer_angles))),
    ))

    ii, jj = np.triu_indices(16, 1)
    pair_count = len(ii)
    rows = np.arange(pair_count)

    def quality(points: np.ndarray) -> float:
        """Return the squared minimum-distance to diameter ratio."""
        delta = points[ii] - points[jj]
        d2 = np.sum(delta * delta, axis=1)
        return float(d2.min() / d2.max())

    def constraints(z: np.ndarray) -> np.ndarray:
        """Enforce unit separation and an explicit common diameter variable."""
        points = z[:-1].reshape(16, 2)
        diameter = z[-1]
        delta = points[ii] - points[jj]
        d2 = np.sum(delta * delta, axis=1)
        return np.concatenate((d2 - 1.0, diameter * diameter - d2))

    def constraint_jacobian(z: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of all pairwise separation and diameter walls."""
        points = z[:-1].reshape(16, 2)
        diameter = z[-1]
        delta = points[ii] - points[jj]
        jac = np.zeros((2 * pair_count, 33), dtype=np.float64)

        jac[rows, 2 * ii] = 2.0 * delta[:, 0]
        jac[rows, 2 * ii + 1] = 2.0 * delta[:, 1]
        jac[rows, 2 * jj] = -2.0 * delta[:, 0]
        jac[rows, 2 * jj + 1] = -2.0 * delta[:, 1]

        jac[pair_count + rows, :32] = -jac[rows, :32]
        jac[pair_count + rows, 32] = 2.0 * diameter
        return jac

    # This is the direct finite packing formulation: maximize the guaranteed
    # separation after fixing it to one, while minimizing the common diameter.
    # It avoids the continuation-wall mismatch of the previous force method.
    def objective(z: np.ndarray) -> float:
        return float(z[-1])

    def objective_jacobian(z: np.ndarray) -> np.ndarray:
        grad = np.zeros(33, dtype=np.float64)
        grad[-1] = 1.0
        return grad

    best = base.copy()
    best_quality = quality(best)
    k = np.arange(16, dtype=np.float64)
    starts = [base]

    # Feasible asymmetric starts help SLSQP reach distinct contact graphs.
    for phase, frequency in ((0.4, 2.71), (1.7, 1.39), (3.1, 3.83)):
        points = base + 0.035 * np.column_stack((
            np.sin(frequency * k + phase),
            np.cos((frequency + 0.91) * k - phase),
        ))
        points -= points.mean(axis=0)
        delta = points[ii] - points[jj]
        points /= np.sqrt(np.min(np.sum(delta * delta, axis=1)))
        starts.append(points)

    # A compact eight-parameter family is searched globally before Cartesian
    # refinement.  Its variables are inner radius, outer radius, alternating
    # outer-radius displacement, shell phase, two paired inner gap modes, and
    # two paired outer angular-offset modes.  It deliberately has no imposed
    # reflection symmetry.
    def shell_points(x: np.ndarray) -> np.ndarray:
        inner_radius, outer_radius, alternating, phase, gi, hi, go, ho = x
        j5 = np.arange(5, dtype=np.float64)
        j10 = np.arange(10, dtype=np.float64)
        inner_gaps = 2.0 * np.pi / 5.0 + gi * np.sin(2.0 * np.pi * j5 / 5.0) + hi * np.cos(2.0 * np.pi * j5 / 5.0)
        inner_angles = np.r_[0.0, np.cumsum(inner_gaps[:-1])]
        outer_angles = (
            phase + 2.0 * np.pi * j10 / 10.0
            + go * np.sin(2.0 * np.pi * j10 / 10.0)
            + ho * np.cos(2.0 * np.pi * j10 / 10.0)
        )
        outer_radii = outer_radius + alternating * ((-1.0) ** j10)
        points = np.vstack((
            [[0.0, 0.0]],
            inner_radius * np.column_stack((np.cos(inner_angles), np.sin(inner_angles))),
            outer_radii[:, None] * np.column_stack((np.cos(outer_angles), np.sin(outer_angles))),
        ))
        points -= points.mean(axis=0)
        delta = points[ii] - points[jj]
        return points / np.sqrt(np.max(np.sum(delta * delta, axis=1)))

    def shell_objective(x: np.ndarray) -> float:
        # Score the explicitly diameter-normalized coordinates using all pairs,
        # rather than relying on SHGO's local solver's stopping status.
        return -quality(shell_points(x))

    bounds = [
        (0.80, 1.30), (1.55, 2.35), (-0.28, 0.28), (0.0, np.pi / 5.0),
        (-0.20, 0.20), (-0.20, 0.20), (-0.20, 0.20), (-0.20, 0.20),
    ]
    try:
        global_result = shgo(
            shell_objective,
            bounds,
            n=192,
            iters=1,
            sampling_method="simplicial",
            minimizer_kwargs={"method": "Nelder-Mead", "options": {"maxiter": 500}},
            options={"f_tol": 1.0e-12},
        )
        shell_candidates = [global_result.x]
        if hasattr(global_result, "xl"):
            shell_candidates.extend(global_result.xl)
        # Only a few objectively best distinct SHGO basins are passed to the
        # more expensive full 32-coordinate constrained refinement.
        ranked = sorted(
            ((quality(shell_points(x)), shell_points(x)) for x in shell_candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        for _, points in ranked[:4]:
            delta = points[ii] - points[jj]
            points = points / np.sqrt(np.min(np.sum(delta * delta, axis=1)))
            starts.append(points)
    except Exception:
        # The established deterministic starts remain valid if a platform has
        # an unusually restrictive SHGO implementation.
        pass

    specification = {
        "type": "ineq",
        "fun": constraints,
        "jac": constraint_jacobian,
    }

    for points in starts:
        delta = points[ii] - points[jj]
        diameter = np.sqrt(np.max(np.sum(delta * delta, axis=1)))
        initial = np.concatenate((points.ravel(), [diameter]))

        result = minimize(
            objective,
            initial,
            jac=objective_jacobian,
            method="SLSQP",
            bounds=[(None, None)] * 32 + [(0.5, None)],
            constraints=specification,
            options={"maxiter": 2500, "ftol": 1.0e-11, "disp": False},
        )

        candidate = result.x[:-1].reshape(16, 2)
        candidate -= candidate.mean(axis=0)
        candidate_quality = quality(candidate)
        if candidate_quality > best_quality:
            best_quality = candidate_quality
            best = candidate

    delta = best[ii] - best[jj]
    diameter = np.sqrt(np.max(np.sum(delta * delta, axis=1)))
    return best / diameter


# EVOLVE-BLOCK-END
