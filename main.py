import numpy as np
from numpy._core.multiarray import promote_types
from src import Config1D, initialize_simulation, vPoisson, plot_results, step


Nufi_fs = None
Nufi_data = None
Nufi_params = Config1D(
    Nx=2**4,  # num. of grid points
    Nv=2**4,  # num. of grid points
    Nx_eval=2**6,  # num. of points when evaluating distribution
    Nv_eval=2**6,  # num. of points when evaluating distribution
    Mass=[1],  # species mass
    Charge=[-1],  # species charge
    Ns=1,  # number of species
    S_name="two_stream",  # simulation case name
    Nt_max=None,  # maximum number of time steps (None => t_end/dt)
    dt=1 / 10,  # time step size
    t_end=30,  # end time of simulation
    plot_freq=10,  # iterations between plotting
    measure_freq=1,  # iterations between measurements
    k=0.5,  # wave number
    eps=1e-3,  # perturbation amplitude
    v0=3,  # electron drift velocity
    gridView=None,  # Dune gridView (None => built from parameters)
    f0=None,
    time=0,
)

# Start grid and fs (type(fs)=np.array)
Nufi_params, Nufi_fs, Nufi_data = initialize_simulation(Nufi_params)

diff_plots = True  # plot diff plots (True), or distribution f (False)
# set type of plot to f_t-fini (False), or f_t(x)-1/sqrt(2pi)v^2exp(-v^2) (True)
_1D_diff_plot = False

if diff_plots:
    if _1D_diff_plot:
        ptype = 0
        finitial = np.array(
            [Nufi_params.fini_v(i) for i in Nufi_params.v_sampling_grid]
        )
    else:
        ptype = 1
        finitial = Nufi_fs
else:
    finitial = None
# Start data
Nufi_data.Efield = vPoisson(Nufi_params, Nufi_fs, Nufi_params.Charge[0])
Nufi_data.Efield_list = np.zeros((Nufi_params.Nx_eval, Nufi_params.Nt_max + 1))
Nufi_data.Efield_list[:, 0] = Nufi_data.Efield
Nufi_data.fs = Nufi_fs

# Make initial plot
plot_results(
    Nufi_params,
    Nufi_data,
    Nufi_fs,
    savedir="plots",
    savename="initial_plot",
    saving=True,
    fini=finitial,
    ptype=ptype,
)
# # ---- Main loop ---- #

Nsamples = 0
time = 0
framenr = 1
for i in range(Nufi_params.Nt_max):
    Nufi_params.it = i
    # print(f"in main loop, i={i}")
    Nufi_params, Nufi_data, Nufi_fs = step(Nufi_params, Nufi_data, Nufi_fs)
    # print("step called succesfully")
    time += Nufi_params.dt
    Nufi_params.time = time
    Nufi_params.time_array.append(time)
    print(f"sim time = {round(Nufi_params.time, 3)}")

    # Plot at frequency
    if i % (Nufi_params.plot_freq) == 0:
        plot_results(
            Nufi_params,
            Nufi_data,
            Nufi_fs,
            savedir="plots/frames",
            savename=f"{framenr}",
            saving=True,
            fini=finitial,
            ptype=ptype,
        )
        framenr += 1

# Plot Final results
plot_results(Nufi_params, Nufi_data, Nufi_fs, savename="finalplot", saving=True)
