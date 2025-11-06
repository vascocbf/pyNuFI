from config import Config1D
from initialize import initialize_simulation
from fields import vPoisson
import numpy as np
from plotting import plot_results
from nufi import step

Nufi_fs = None
Nufi_data = None
Nufi_params = Config1D(
    Nsample=[2**6, 2**6],        # sample grid points
    Nmap=[2**6, 2**6],           # map grid points
    Nx=None,                      # optional, can leave as None
    Nv=None,                      # optional, can leave as None
    Mass=[1],                     # species mass
    Charge=[-1],                  # species charge
    Ns=1,                         # number of species
    S_name="two_stream",          # simulation case name
    Mr=1,                         # mass ratio for ions
    Nt_max=4000,                  # maximum number of time steps
    dt=1/20,                       # time step size
    dt_save=10,                    # save interval (not used)
    t_end=40,                      # end time of simulation
    plot_freq=5,                  # iterations between plotting
    measure_freq=1,                # iterations between measurements
    k=0.5,                         # wave number
    eps=1e-2,                      # perturbation amplitude
    v0=3                           # electron drift velocity
)


# Start grid and fs 
Nufi_params, Nufi_fs, Nufi_data = initialize_simulation(Nufi_params)
 
# Start data
Nufi_data.Efield = vPoisson(Nufi_fs, Nufi_params.grids, Nufi_params.Charge)
Nufi_data.Efield_list = np.zeros((Nufi_params.grids[0].Nx, Nufi_params.Nt_max+1))
Nufi_data.Efield_list[:,0] = Nufi_data.Efield
Nufi_data.time = 0 
Nufi_data.fs = Nufi_fs

# Make initial plot 
plot_results(Nufi_params, Nufi_data, Nufi_fs)


# ---- Main loop ---- # 

Nsamples = 0
time = 0 

for i in range(Nufi_params.Nt_max):

    Nufi_params.it = i
    
    Nufi_params, Nufi_data, Nufi_fs = step(Nufi_params, Nufi_data, Nufi_fs)
    
    time += Nufi_params.dt
    Nufi_params.time = time
    Nufi_params.time_array.append(time)

    # plot loop

    #
plot_results(Nufi_params, Nufi_data, Nufi_fs)
