import numpy as np
from dune.common import FieldVector

def eval_f(params):
    """
    Given a gridFunction, return evaluation of f on spline
    return np.array with shape (Nx_eval, Nv_eval)
    """
    Nx_eval = params.Nx_eval
    Nv_eval = params.Nv_eval

    x_vals = np.linspace(0, params.Lx, num=Nx_eval)
    v_vals = np.linspace(-params.Lv, params.Lv, num=Nv_eval)
    fini = params.fini # a Dune gridFunction

    f = np.zeros((Nx_eval, Nv_eval))
    a = FieldVector([0,0])
    for i, x in enumerate(x_vals):
        a[0]=x
        for j, v in enumerate(v_vals):
            a[1]=v
            f[i,j] = fini(a)

    return f

def compute_density(fs, dv):
    return np.sum(fs * dv, axis=1)

def vPoisson(fs, grids, charge):
    """
    Solve 1D Poisson equation for the electric field given the distribution function.

    Parameters
    ----------
    fs : ndarray, shape (Nv, Nx, Ns)
        Distribution function for all species
    grids : list of Grid
        List of Grid instances, must have dv, kx, kx2
    charge : list or ndarray, length Ns
        Charges for each species

    Returns
    -------
    Efield : ndarray, shape (Nx,)
        Electric field at each spatial point
    """
    Ns = len(grids)
    rho = np.zeros(grids[0].Nx)
    
    # Compute total charge density
    for s in range(Ns):
        rho += charge[s] * compute_density(fs[:, :, s], grids[s].dv)
    
    kx = grids[0].kx
    K2 = np.copy(grids[0].kx2)
    
    # Solve Poisson in Fourier space
    b = np.fft.fft(1-rho)
    phi_fft = -b / K2
    phi_fft[0] = 0  # set mean to zero (zero mode)
    
    # Compute electric field: E = -dphi/dx
    dphi_dx_h = -1j * phi_fft * kx
    Efield = -np.real(np.fft.ifft(dphi_dx_h))  # 1D field
    
    return Efield
