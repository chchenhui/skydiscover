# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Return a 16-point triangular-lattice cluster with three alternate corners removed.

    The underlying triangular lattice has nearest-neighbor spacing one.  Taking
    a radius-two hexagonal cluster gives 19 points; removing three alternating
    extreme vertices leaves 16 points, retains minimum separation 1, and has
    maximum pairwise distance sqrt(13).
    """
    # Axial coordinates (q, r) for a triangular lattice.  Cartesian mapping:
    # x = q + r/2, y = sqrt(3)*r/2.
    axial = np.array(
        [
            [0, -2], [1, -2],
            [-1, -1], [0, -1], [1, -1], [2, -1],
            [-2, 0], [-1, 0], [0, 0], [1, 0],
            [-2, 1], [-1, 1], [0, 1], [1, 1],
            [-2, 2], [-1, 2],
        ],
        dtype=np.float64,
    )

    points = np.empty((16, 2), dtype=np.float64)
    points[:, 0] = axial[:, 0] + 0.5 * axial[:, 1]
    points[:, 1] = 0.5 * np.sqrt(3.0) * axial[:, 1]
    return points


# EVOLVE-BLOCK-END
