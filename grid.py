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
    kv: np.ndarray = None
    kv2: np.ndarray = None

    # periodicity and weights
    Vperiodic: np.ndarray = None
    Weights: np.ndarray = None

    method: str = "spline"

# ------- Grid funcs ------- #

def velocity_periodicfication(params, bump_transition_width=None, type_="exp"):
    r""" (Copied from matlab code)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% first calculate "vorticity" such that global vorticity omega = 0 
    % (u_1,u_2) = (g(v), d/dx \phi)
    % omega = d/dx u_2 - d/dv u_1 = d^2/(dx)2 \phi(x) - d/dv g(v) == 0 
    % assume phi(x)=0 
    % if g(v) = v 
    % omega = d/dv g(v) = 1 != 0 ):
    % so we do the following:
    % look for a function h(v) = d/dv g(v) that keeps global omega=0, 
    % but is 1 close to the
    % origin and negative on the boundaries (= heaviside function)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    
    Lv = params.Lv
    v = params.v
    kv = np.copy(params.kv)
    kv2 = kv**2
    kv2[0] = 1.0 

    # --- handle bump width and type ---
    if bump_transition_width is None:
        b = 0.2 * Lv
    else:
        b = bump_transition_width

    # --- choose type of transition ---
    if type_ == "tanh":
        a = 0.4
        sigma = 0.5 - 0.5 * np.tanh(2 * np.pi * (np.abs(v) - a * 2 * Lv) / b)
        #weights = None  # not explicitly defined in this case
    elif type_ == "exp":
        def nu(x):
            y = np.zeros_like(x)
            mask = np.abs(x) < 1
            y[mask] = np.exp(-1.0 / (1 - x[mask] ** 2))
            return y

        sigma = 1 - nu((np.abs(v) - Lv) / b)
        #weights = nu((np.abs(v) - Lv) / b)
    else:
        raise ValueError(f"Unknown type: {type_}")

    # --- normalize h ---
    h = np.copy(sigma)
    h = h - np.mean(h)
    h = h / np.max(h)

    # --- compute periodic velocity ---
    intu_hat = -np.fft.fft(h) / kv2
    #intu = np.fft.ifft(intu_hat)  # complex array, but we take symmetric part below
    v_periodic = np.real(np.fft.ifft(1j * intu_hat * kv))

    return v_periodic, sigma

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
    kv = np.fft.fftshift((np.pi/Lv) * np.arange(-Nv//2, Nv//2))
    kv2 = kv**2
    kv2[0] = 1.0


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
        kv=kv,
        kv2=kv2,
        Vperiodic=None,
        Weights=None,
        method="spline"
    )
    
    # Periodic velocity
    vper, sigma = velocity_periodicfication(grid)
    _, Vperiodic = np.meshgrid(x, vper, indexing='ij')
    _, Weights = np.meshgrid(x, np.abs(sigma) < 1e-12, indexing='ij')
    grid.Vperiodic = Vperiodic
    grid.Weights = Weights

    return grid
