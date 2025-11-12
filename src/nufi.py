from .fields import vPoisson
from scipy.interpolate import CubicSpline 
import numpy as np
        

def NuFi(params, data, fs):
    """
    Single NuFi time step update for all species.
    
    Inputs:
        params - configuration object containing grids, charges, masses, etc.
        fs     - distribution function array (Nx x Nv x Ns)
        
    Returns:
        fs     - updated distribution function
        params - updated parameters (Efield, Efield_list)
    """
    iT = params.it + 1
    dt = params.dt
    Ns = params.Ns

    for s in range(Ns):
        grid = params.grids[s]
        charge_s = params.Charge[s]
        mass_s = params.Mass[s]
        X, V = sympl_flow_Half(
            n=iT,
            dt=dt,
            X=grid.X,
            V=grid.V,
            Efield_list=data.Efield_list,
            grid=grid,
            params=params,
            charge=charge_s,
            mass=mass_s
        )

        # Update distribution function
        fini = params.fini[s]
        fs[:, :, s] = fini(X, V)


    # Compute electric field
    Efield = vPoisson(fs, params.grids, params.Charge)
    # Add external field
    #Efield += compute_external_Efield(params, params.grids[0].x, params.time + dt)

    # Update parameters
    data.Efield = Efield
    data.Efield_list[:, iT] = Efield

    return params, data, fs


def sympl_flow_Half(n, dt, X, V, Efield_list, grid, params, charge, mass):
    """
    Symplectic flow for half time step in NuFi method.
    
    Inputs:
        n      : step number
        dt     : time step
        X, V   : position and velocity arrays
        Efield : electric field array (1D)
        grid   : grid object
        params : simulation parameters
        charge : particle charge
        mass   : particle mass
    Outputs:
        X, V   : updated position and velocity arrays
    """
    
    Efield_list_normed = Efield_list * (charge/mass)
    if n == 0:
        return X, V

    # Velocity field (positions advance with current velocities)
    def Ux(X_, V_):
        return V_

    # Acceleration field (velocity update from electric field)
    def Uv(X_, E):
    # Interpolate E(x) onto the 2D X_ grid; returns shape like X_
        return interp1d_periodic(X_, params.grids[0].x, E)

    # Full steps if n > 2
    if n > 1:
        for i in range(n - 1):
            X = X - dt * Ux(X, V)
            X = wrap_periodic(X, params.grids[0].x)   # <-- add this line
            # Use the corresponding past Efield if needed; here we just use current
            V = V + dt * Uv(X, Efield_list_normed[:, n - i])

    # Final half step
    X = X - dt * Ux(X, V)
    X = wrap_periodic(X, params.grids[0].x)   # <-- add this line
    V = V + 0.5 * dt * Uv(X, Efield_list_normed[:, 0])

    return X, V

def wrap_periodic(X, xgrid):
    dx = xgrid[1] - xgrid[0]
    L  = dx * len(xgrid)
    x0 = xgrid[0]
    return (X - x0) % L + x0

def interp1d_periodic(xq, xgrid, Fgrid):
    """
    Periodic cubic interpolation of Fgrid(xgrid) evaluated at xq.
    - xgrid: shape (Nx,), strictly periodic (no duplicated endpoint).
    - Fgrid: shape (Nx,)
    - xq   : any shape (...), returns same shape as xq
    """
    xgrid = np.asarray(xgrid)
    Fgrid = np.asarray(Fgrid)
    xq    = np.asarray(xq)

    dx = xgrid[1] - xgrid[0]
    L  = dx * len(xgrid)
    x0 = xgrid[0]

    # wrap queries into [x0, x0+L)
    xq_mod = (xq - x0) % L + x0

    # append duplicate endpoint for the spline only
    x_ext = np.concatenate([xgrid, [x0 + L]])
    F_ext = np.concatenate([Fgrid, [Fgrid[0]]])

    spline = CubicSpline(x_ext, F_ext, bc_type='periodic')
    Fq = spline(xq_mod)            # vectorized; same shape as xq
    return Fq

def step(params, data, fs):
    """
    Time step for simulation, update parameters, field, and fs
    """

    params, data, fs = NuFi(params, data, fs)
    data.Efield_list[:,params.it] = data.Efield

    return params, data, fs
