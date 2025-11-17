import numpy as np
from dataclasses import dataclass
from scipy.sparse import spdiags

# ------- Grid Class ------- #

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

    method: str = "spline"

# ------- Grid funcs ------- #
def make_periodic_grid(Lx, Lv, Nx, Nv):
    # 1D grids
    x = np.arange(Nx) * Lx / Nx
    v = np.arange(Nv) * 2*Lv / Nv - Lv  # from -Lv to Lv
    dx = x[1] - x[0]
    dv = v[1] - v[0]

    # Meshgrid for phase space
    X, V = np.meshgrid(x, v, indexing='ij')
    
    # data must have shape (n_diagonals, N)
    Shape_x = np.vstack((
        -np.ones(Nx), 
        np.zeros(Nx),  
        np.ones(Nx)    
    ))  # shape (3, Nx)

    Shape_v = np.vstack((
        -np.ones(Nv), 
        np.zeros(Nv),  
        np.ones(Nv)    
    ))  # shape (3, Nv)
    
    Dx = spdiags(Shape_x, [-1, 0, 1], Nx, Nx).toarray()
    Dv = spdiags(Shape_v, [-1, 0, 1], Nv, Nv).toarray()
    
    # Apply periodic BCs (connect first and last)
    Dx[0, -1] = -1
    Dx[-1, 0] = 1
    Dv[0, -1] = -1
    Dv[-1, 0] = 1
    
    # Scale by spacing
    Dx /= (2 * dx)
    Dv /= (2 * dv)
    # Fourier wavenumbers
    kx = np.fft.fftshift((2*np.pi/Lx) * np.arange(-Nx//2, Nx//2))
    kx2 = kx**2
    kx2[0] = 1.0


    grid = Grid(
        sample=None,
        map=None,
        idx_sample_to_map=(None, None),
        x=x,
        v=v,
        X=X,
        V=V,
        Xsample_grid=X,
        Vsample_grid=V,
        size=X.shape,
        size_sample_grid=X.shape,
        dom=[0, -Lv, Lx-dx, Lv-dv],
        dx=dx,
        dv=dv,
        Lx=Lx,
        Lv=Lv,
        Nx=Nx,
        Nv=Nv,
        Dx=Dx,
        Dv=Dv,
        kx=kx,
        kx2=kx2,
        method="spline"
    )

    return grid
