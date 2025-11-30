from .data import DataStorage
import numpy as np
from .grid import make_periodic_grid, Grid
from .fields import eval_f


def initialize_simulation(params):
    """
    Initialize:
    grids,
    distribution functions,
    and storage arrays.
    """
    data = DataStorage()

    # use Nx and Nv if defined
    N = [params.Nx, params.Nv]

    # Create sample and map grids
    grid = make_periodic_grid(params.Lx, params.Lv, N[0], N[1])

    # Create Grid dataclass instance
    grid_obj = Grid(
        dv=grid.dv,
        Lx=grid.Lx,
        Lv=grid.Lv,
        Nx=grid.Nx,
        Nv=grid.Nv,
    )
    params.grids.append(grid_obj)

    # Ensure Nt_max fits t_end
    if params.Nt_max > params.t_end / params.dt:
        params.Nt_max = int(np.ceil(params.t_end / params.dt))

    f = eval_f(params, params.x_sampling_grid, params.v_sampling_grid)

    return params, f, data
