import matplotlib.pyplot as plt
from .fields import compute_density
import numpy as np

def plot_results(params, data, fs, initial_plot=False):
    """
    Plot the distribution function, electric field, density, and field energy evolution.
    """
    Ns = params.Ns  # typically 1
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # === Plot 1: distribution function ===
    ax = axes[0]
    X = params.grids[0].Xsample_grid
    V = params.grids[0].Vsample_grid
    im = ax.pcolormesh(X, V, fs[:, :, 0], shading='auto')
    ax.set_title(r"$f_\mathrm{" + params.S_name[0] + "}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    fig.colorbar(im, ax=ax)

    # === Plot 2: electric field ===
    ax = axes[1]
    x = params.grids[0].x
    Efield = data.Efield
    ax.plot(x, Efield)
    ax.set_xlim([x[0], x[-1]])
    ax.set_title(r"$E$" + f" at t = {data.time:.2f}")
    ax.set_xlabel(r"$x$")
    ax.grid(True)

    # === Plot 3: 1 - density ===
    ax = axes[2]
    density = compute_density(fs, params.grids[0].dv)
    ax.plot(x, 1 - density)
    ax.set_xlim([x[0], x[-1]])
    ax.set_title(r"$1 - \rho$")
    ax.set_xlabel(r"$x$")
    ax.grid(True)

    # === Plot 4: field energy evolution ===
    if initial_plot == False:
        ax = axes[3]
        maxE = 0.5 * np.sum(data.Efield_list**2, axis=0)
        ts = np.arange(len(maxE)) * params.dt 
        ax.semilogy(ts, maxE)
        ax.set_title(r"$\frac{1}{2}\sum_x E^2$ vs time")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("Energy (log scale)")
        ax.grid(True)

    plt.tight_layout()
#   plt.pause(0.01)
#    plt.show()
    plt.savefig("plot.png")
