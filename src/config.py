import numpy as np
from dataclasses import dataclass
from dune.fem.function import gridFunction
from dune.grid import cartesianDomain, yaspGrid
from math import pi, cos, exp, sqrt
from typing import Any, Optional


@dataclass
class Config1D:
    """
    Contains:
    - simulation parameters
    - gridView
    - gridFunction
    - grid to sample points from (a np.linspace)
    - wave numbers array (np.fft.fftshift)
    """

    # grid settings
    Nx: int = 2**6
    Nv: int = 2**6
    Nx_eval: int = 2**6
    Nv_eval: int = 2**6
    Mass: Optional[list] = None
    Charge: Optional[list] = None
    Ns: int = 1  # num of species
    S_name: str = "electrons"
    Nt_max: Optional[int] = None  # None => t_end/dt

    # spline settings
    order: int = 3  # spline order

    # sim settings
    dt: float = 0.1  # time step
    t_end: int = 20
    plot_freq: int = 5  # iterations between plots
    measure_freq: int = 1  # iterations between measurements

    k: float = 0.5  # wave nr
    eps: float = 1e-2  # perturbation amplitude

    v0: float = 3  # electron drift velocity

    grids: Optional[list] = None

    it: int = 0  # simulation iteration tic
    time: float = 0
    time_array: Optional[list] = None

    # Dune objects, definition at __post_init__
    gridDomain: Any = None
    gridView: Any = None
    f0: Any = None
    fini: Any = None

    x_sampling_grid: Optional[np.ndarray] = None
    v_sampling_grid: Optional[np.ndarray] = None

    def __post_init__(self):
        self.Lx = 2 * np.pi / self.k  # spatial domain length
        self.Lv = 2 * np.pi  # velocity domain length

        if self.grids is None:
            self.grids = []
        # grid settings
        if self.Mass is None:
            self.Mass = [1]
        if self.Charge is None:
            self.Charge = [-1]
        if self.time_array is None:
            self.time_array = []
        if self.Nt_max is None:
            self.Nt_max = int(self.t_end / self.dt) + 1

        # Dune objects
        if self.gridDomain is None:
            self.gridDomain = cartesianDomain(
                [0, -self.Lv],
                [self.Lx, self.Lv],
                [self.Nx - 1, self.Nv - 1],
                periodic=[True, False],
            )
        if self.gridView is None:
            self.gridView = yaspGrid(self.gridDomain, dimgrid=2)
        if self.f0 is None:
            self.f0 = (
                lambda x: (1 + self.eps * cos(self.k * x[0]))
                * x[1] ** 2
                / (sqrt(2 * pi))
                * exp(-(x[1] ** 2) / 2)
            )

        # a dune.fem gridFunction
        self.fini = gridFunction(
            self.f0,
            gridView=self.gridView,
            name="fini",
            order=self.order,
        )
        self.dv = self.Lv / self.Nv_eval

        # Fourier wavenumbers
        self.kx = np.fft.fftshift(
            (2 * np.pi / self.Lx) * np.arange(-self.Nx_eval // 2, self.Nx_eval // 2)
        )
        self.kx2 = self.kx**2
        self.kx2[0] = 1.0

        if self.x_sampling_grid is None:
            self.x_sampling_grid = np.linspace(start=0, stop=self.Lx, num=self.Nx_eval)
        if self.v_sampling_grid is None:
            self.v_sampling_grid = np.linspace(
                start=-self.Lv, stop=self.Lv, num=self.Nv_eval
            )
