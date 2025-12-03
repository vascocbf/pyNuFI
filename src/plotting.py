import matplotlib.pyplot as plt
from .fields import compute_density
import numpy as np


def plot_results(
    params, data, fs, savedir="plots", savename="plot", saving=False, fini=None, ptype=0
):
    """
    Plot the distribution function, electric field, density, and field energy evolution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    X = params.x_sampling_grid
    V = params.v_sampling_grid
    # === Plot 1: distribution function ===
    ax = axes[0]
    if ptype == 0:
        if fini is not None:
            fs_plot = fs[1] - fini
            im = ax.scatter(V, fs_plot)
            ax.set_title("f_t - 1/sqrt(2pi) v_0^2 exp(-v_0^2)")
        else:
            im = ax.pcolormesh(X, V, fs.T, shading="auto")
            ax.set_title(r"$f_\mathrm{" + params.S_name[0] + "}$")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$v$")
            fig.colorbar(im, ax=ax)
    elif ptype == 1:
        if fini is not None:
            fs_plot = fs - fini
            im = ax.pcolormesh(X, V, fs_plot.T, shading="auto")
        else:
            im = ax.pcolormesh(X, V, fs.T, shading="auto")
        ax.set_title(r"$f_\mathrm{" + params.S_name[0] + "}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$v$")
        fig.colorbar(im, ax=ax)

    # === Plot 2: electric field ===
    ax = axes[1]
    Efield = data.Efield
    time = round(params.time, 2)
    ax.plot(X, Efield)
    ax.set_title(r"$E$" + f" at t = {time}")
    ax.set_xlabel(r"$x$")
    ax.grid(True)

    # === Plot 3: 1 - density ===
    ax = axes[2]
    density = compute_density(fs, V)
    ax.plot(X, 1 - density)
    ax.set_title(r"$1 - \rho$" + f" at t = {time}")
    ax.set_xlabel(r"$x$")
    ax.grid(True)

    # === Plot 4: field energy evolution ===
    ax = axes[3]
    maxE = 0.5 * np.sum(data.Efield_list**2, axis=0)
    ts = np.arange(len(maxE)) * params.dt
    ax.semilogy(ts, maxE)
    ax.set_title(r"$\frac{1}{2}\sum_x E^2$ vs time")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("E")
    ax.grid(True)

    plt.tight_layout()
    #   plt.pause(0.01)
    #    plt.show()
    if saving:
        plt.savefig(f"./{savedir}/{savename}.png")
    plt.close(fig)
