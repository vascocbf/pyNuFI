import matplotlib.pyplot as plt
from .fields import compute_density

def plot_results(params, data, fs, time=None):
    """
    Plot the distribution function and electric field.

    Parameters
    ----------
    params : object
        Simulation parameters, must have `Ns`, `grids`, `species_name`, and `Efield`.
    fs : ndarray, shape (Nv, Nx, Ns)
        Distribution function for all species
    """
    Ns = params.Ns
    total_plots = Ns + 2 
    fig, axes = plt.subplots(total_plots, 1, figsize=(8, 3*(total_plots)))

    # Ensure axes is always a list
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # flatten if it’s an array

    # Plot distribution function for each species
    for s in range(Ns):
        ax = axes[s]
        X = params.grids[s].Xsample_grid
        V = params.grids[s].Vsample_grid
        im = ax.pcolormesh(X, V, fs[:, :, s], shading='auto')
        ax.set_title(r"$f_\mathrm{" + params.S_name[s] + "}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$v$")
        fig.colorbar(im, ax=ax)

    # Plot electric field
    ax = axes[Ns]
    x = params.grids[0].x
    Efield = data.Efield
    ax.plot(x, Efield)
    ax.set_xlim([x[0], x[-1]])
    ax.set_title("$E$" + (f" at t = {time:.2f}" if time is not None else ""))
    ax.set_xlabel(r"$x$")
    ax.grid(True)

    # Plot 1 - density
    ax = axes[Ns+1]
    density = compute_density(fs, params.grids[0].dv)
    ax.plot(x, 1 - density)
    ax.set_xlim([x[0], x[-1]])
    ax.set_title(r"$1 - \rho$")
    ax.set_xlabel(r"$x$")
    ax.grid(True)
    
    plt.tight_layout()
    plt.pause(0.01)
    plt.show()
