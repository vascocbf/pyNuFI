import numpy as np
from dataclasses import dataclass
from dune.grid import structuredGrid

# ------- Grid Class ------- #

@dataclass
class Grid:
    # spacing
    dv: float = 0.0

    # domain lengths
    Lx: float = 0.0
    Lv: float = 0.0

    # number of points
    Nx: int = 0
    Nv: int = 0

    # wavenumbers
    kx: np.ndarray = None
    kx2: np.ndarray = None

    method: str = "spline"

    # Dune grid, a structuredGrid object
    gridView = None

# ------- Grid funcs ------- #
def make_periodic_grid(Lx, Lv, Nx, Nv):
    # 1D grids
    dv = Lv/Nv
    
    # Dune gridView
    gridView = structuredGrid([0,-Lv], [Lx,Lv], [Nx-1,Nv-1])
    
    # Fourier wavenumbers
    kx = np.fft.fftshift((2*np.pi/Lx) * np.arange(-Nx//2, Nx//2))
    kx2 = kx**2
    kx2[0] = 1.0

    grid = Grid(
        dv=dv,
        Lx=Lx,
        Lv=Lv,
        Nx=Nx,
        Nv=Nv,
        kx=kx,
        kx2=kx2,
        method="spline",
        gridView=gridView,
    )

    return grid
