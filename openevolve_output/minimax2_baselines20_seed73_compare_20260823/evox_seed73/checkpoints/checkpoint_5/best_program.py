# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """Return a compact 16-point subset of a triangular lattice.

    The points have nearest-neighbor distance 1.  They are selected from the
    center and first two shells of the hexagonal lattice, with three outer
    vertices removed to reduce the diameter from 4 to sqrt(13).
    """
    h = np.sqrt(3.0) / 2.0

    points = np.array([
        # Center
        [0.0, 0.0],

        # First hexagonal shell: all six points at distance 1
        [1.0, 0.0],
        [0.5, h],
        [-0.5, h],
        [-1.0, 0.0],
        [-0.5, -h],
        [0.5, -h],

        # Nine carefully chosen second-shell points
        [1.0, 2.0 * h],
        [-2.0, 0.0],
        [1.0, -2.0 * h],
        [1.5, h],
        [0.0, 2.0 * h],
        [-1.5, h],
        [-1.5, -h],
        [0.0, -2.0 * h],
        [1.5, -h],
    ], dtype=np.float64)

    return points


# EVOLVE-BLOCK-END
