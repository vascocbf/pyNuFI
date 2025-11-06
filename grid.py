
import numpy as np
from scipy.sparse import spdiags
from dataclasses import dataclass
import numpy.fft as fft


@dataclass
class Grid:
    # core grids
    sample: any
    map: any

    # index mapping from sample to map
    idx_sample_to_map: tuple

    # phase space grids
    x: np.ndarray
    v: np.ndarray
    X: np.ndarray
    V: np.ndarray
    Xsample_grid: np.ndarray
    Vsample_grid: np.ndarray

    # grid sizes
    size: tuple
    size_sample_grid: tuple
    dom: any = None

    # spacing
    dx: float = 0.0
    dv: float = 0.0

    # domain lengths
    Lx: float = 0.0
    Lv: float = 0.0

    # number of points
    Nx: int = 0
    Nv: int = 0

    # spacing arrays for derivatives
    Dx: np.ndarray = None
    Dv: np.ndarray = None

    # wavenumbers
    kx: np.ndarray = None
    kx2: np.ndarray = None
    kv: np.ndarray = None
    kv2: np.ndarray = None

    # periodicity and weights
    Vperiodic: np.ndarray = None
    Weights: np.ndarray = None

    method: str = "spline"
