from .fields import vPoisson, eval_f, wrap_periodic, E_spline
from math import pi, sqrt, exp, cos
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

    Efield, fs, V_new = Half_flow(
        n=iT,
        dt=dt,
        X=params.x_sampling_grid,
        V=params.v_sampling_grid,
        Efield_list=data.Efield_list,
        params=params,
        charge=params.Charge[0],
        mass=params.Mass[0],
    )

    # Add external field
    # Efield += compute_external_Efield(params, params.grids[0].x, params.time + dt)

    # Update parameters
    data.Efield = Efield
    data.Efield_list[:, iT] = Efield

    return params, data, fs, V_new


def new_eval_f(params, x, v):
    return (
        (1 + params.eps * cos(params.k * x)) * v**2 / (sqrt(2 * pi)) * exp(-(v**2) / 2)
    )


def Half_flow(n, dt, X, V, Efield_list, params, charge, mass):
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
        E, f   : updated Efield and distribution
    """

    Efield_list_normed = Efield_list * (charge / mass)
    if n == 0:
        return X, V

    # Velocity field (positions advance with current velocities)
    def Ux(X_, V_):
        return V_

    # Acceleration field (velocity update from electric field)
    def Uv(X_, E):
        # Interpolate E(x) onto the X_ grid; returns shape like X_
        # x_mod = wrap_periodic(X_, params.x_sampling_grid)
        spline = E_spline(params, X_, params.x_sampling_grid, E)

        return spline(X_)

    # Full steps if n > 2
    if n > 1:
        for i in range(n - 1):
            X = X - dt * Ux(X, V)
            X = wrap_periodic(X, params.x_sampling_grid)
            # Use the corresponding past Efield if needed; here we just use current
            V = V + dt * Uv(X, Efield_list_normed[:, n - i])

    # Final half step
    X = X - dt * Ux(X, V)
    X = wrap_periodic(X, params.x_sampling_grid)
    V = V + 0.5 * dt * Uv(X, Efield_list_normed[:, 0])

    f_new = eval_f(params, X, V)
    # x, v = np.meshgrid(X, V, indexing="ij")
    # f_new = (params, x, v)
    E_new = vPoisson(params, f_new, charge)
    return E_new, f_new, V


def step(params, data, fs):
    """
    Time step for simulation, update parameters, field, and fs
    """

    params, data, fs, V_new = NuFi(params, data, fs)
    data.Efield_list[:, params.it] = data.Efield

    return params, data, fs, V_new
