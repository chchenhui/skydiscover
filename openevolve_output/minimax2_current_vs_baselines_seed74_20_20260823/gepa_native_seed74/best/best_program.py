# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Return a unit-spacing triangular-lattice subset with diameter sqrt(13)."""
    sqrt3 = np.sqrt(3.0)

    # This is the radius-two hexagonal lattice ball with three suitably chosen
    # extreme points omitted.  All nearest-neighbor distances remain 1, while
    # every remaining pair has squared distance at most 13.
    axial = np.array([
        [0, 0],
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1],
        [2, -1], [2, -2], [1, -2],
        [-1, -1], [-2, 0], [-2, 1],
        [-1, 2], [0, 2], [1, 1],
    ], dtype=float)

    points = np.empty((16, 2), dtype=float)
    points[:, 0] = axial[:, 0] + 0.5 * axial[:, 1]
    points[:, 1] = 0.5 * sqrt3 * axial[:, 1]
    return points


# EVOLVE-BLOCK-END
