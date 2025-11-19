import numpy as np
from dune.common import FieldVector

def eval_f(params, x_vals, v_vals):
    """
    Parameters: Nufi_params, x array to eval, v array to eval
    return evaluation of f on spline
    return type np.array with shape (Nx_eval, Nv_eval)
    """
    f = np.zeros((params.Nx_eval, params.Nv_eval))
    a = FieldVector([0,0])
    for i, x in enumerate(x_vals):
        for j, v in enumerate(v_vals):
            a[0]=x
            a[1]=v
            f[i,j] = params.fini(a)

    return f

def compute_density(fs, dv):
    return np.sum(fs * dv, axis=1)

def vPoisson(params, fs, charge):
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
    rho = np.zeros(params.Nx_eval)
    
    # Compute total charge density
    rho += charge * compute_density(fs, params.dv)
    
    kx = params.kx
    K2 = np.copy(params.kx2)
    
    # Solve Poisson in Fourier space
    b = np.fft.fft(1-rho)
    phi_fft = -b / K2
    phi_fft[0] = 0  # set mean to zero (zero mode)
    
    # Compute electric field: E = -dphi/dx
    dphi_dx_h = -1j * phi_fft * kx
    Efield = -np.real(np.fft.ifft(dphi_dx_h))  # 1D field
    
    return Efield
