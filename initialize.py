from data import DataStorage
import numpy as np
from grid import make_periodic_grid, Grid

def initialize_simulation(params):
    """
    Initialize grids, distribution functions, and storage arrays.
    """
    data = DataStorage()
    
    for s in range(params.Ns):
        # Use separate sample and map grids
        Nsample = params.Nsample
        Nmap = params.Nmap

        # Expand scalar values to lists
        if np.isscalar(Nsample):
            Nsample = [Nsample, Nsample]
        if np.isscalar(Nmap):
            Nmap = [Nmap, Nmap]

        # Validate grids
        if any(np.array(Nsample) % np.array(Nmap) != 0):
            raise ValueError("Nsample must be a multiple of Nmap for each dimension")
        if any(np.array(Nmap) > np.array(Nsample)):
            raise ValueError("Nmap must not be bigger than Nsample")

        # Create sample and map grids
        grid_sample = make_periodic_grid(params.Lx, params.Lv, Nsample[0], Nsample[1])
        grid_map = make_periodic_grid(params.Lx, params.Lv, Nmap[0], Nmap[1])
        grid_sample.method = "spline"
        grid_map.method = "spline"

        # Create index mapping from sample grid to map grid
        ratio_x = Nsample[0] // Nmap[0]
        ratio_v = Nsample[1] // Nmap[1]
        idx_sample_to_map = (
            np.arange(0, Nsample[0], ratio_x),
            np.arange(0, Nsample[1], ratio_v)
        )

        # Create Grid dataclass instance
        grid_obj = Grid(
            sample=grid_sample,
            map=grid_map,
            idx_sample_to_map=idx_sample_to_map,
            x=grid_sample.x,
            v=grid_sample.v,
            X=grid_sample.X,
            V=grid_sample.V,
            Xsample_grid=grid_sample.X,
            Vsample_grid=grid_sample.V,
            size=grid_sample.size,
            size_sample_grid=grid_sample.size_sample_grid,
            dom=getattr(grid_sample, "dom", None),
            dx=grid_sample.dx,
            dv=grid_sample.dv,
            Lx=grid_sample.Lx,
            Lv=grid_sample.Lv,
            Nx=grid_sample.Nx,
            Nv=grid_sample.Nv,
            Dx=grid_sample.Dx,
            Dv=grid_sample.Dv,
            kx=grid_sample.kx,
            kx2=grid_sample.kx2,
            kv=grid_sample.kv,
            kv2=grid_sample.kv2,
            Vperiodic=grid_sample.Vperiodic,
            Weights=grid_sample.Weights
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

    # Save parameters for time stepping
    if hasattr(params, "dt_save"):
        dt_save = params.dt_save
        dit_save = dt_save / params.dt
        params.dit_save = dit_save
        if not dit_save.is_integer() or dt_save < params.dt:
            raise ValueError("dt_save is not a proper divisor of dt")
        Nsamples = int(params.Nt_max / dit_save)
        data.fs = np.zeros(params.grids[0].size_sample_grid + (Nsamples, params.Ns))
        data.Efield = np.zeros((params.grids[0].Nx, Nsamples))
        data.time = dt_save * np.arange(1, Nsamples + 1)
    else:
        params.dit_save = params.Nt_max + 2
        data = None

    # Default plotting and measurement frequencies
    if not hasattr(params, "plot_freq") or params.plot_freq == 0:
        params.plot_freq = params.Nt_max
    if not hasattr(params, "measure_freq") or params.measure_freq == 0:
        params.measure_freq = params.Nt_max

    return params, fs, data
