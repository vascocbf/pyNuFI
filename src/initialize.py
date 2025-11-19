from .data import DataStorage
import numpy as np
from .grid import make_periodic_grid, Grid

def initialize_simulation(params):
    """
    Initialize:
    grids, 
    distribution functions,
    and storage arrays.
    """
    data = DataStorage()
    
    for s in range(params.Ns):
        # use Nx and Nv if defined
        N = [params.Nx, params.Nv]

        # Create sample and map grids
        grid = make_periodic_grid(params.Lx, params.Lv, N[0], N[1])

        # Create Grid dataclass instance
        grid_obj = Grid(
            x=grid.x,
            v=grid.v,
            X=grid.X,
            V=grid.V,
            Xsample_grid=grid.X,
            Vsample_grid=grid.V,
            dx=grid.dx,
            dv=grid.dv,
            Lx=grid.Lx,
            Lv=grid.Lv,
            Nx=grid.Nx,
            Nv=grid.Nv,
            kx=grid.kx,
            kx2=grid.kx2,
        )
        params.grids.append(grid_obj)

    # Ensure Nt_max fits t_end
    if params.Nt_max > params.t_end / params.dt:
        params.Nt_max = int(np.ceil(params.t_end / params.dt))

    # Initialize distribution functions
    Nx = params.grids[0].Xsample_grid.shape[0]
    Nv = params.grids[0].Vsample_grid.shape[0]
    fs = np.zeros((Nx, Nv, params.Ns))

    for s in range(params.Ns):
        fini_func = params.fini[s]
        fs[:, :, s] = fini_func(params.grids[s].Xsample_grid, params.grids[s].Vsample_grid)

    # Default plotting and measurement frequencies
    if not hasattr(params, "plot_freq") or params.plot_freq == 0:
        params.plot_freq = params.Nt_max
    if not hasattr(params, "measure_freq") or params.measure_freq == 0:
        params.measure_freq = params.Nt_max

    return params, fs, data
