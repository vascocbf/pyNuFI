import numpy as np
from dataclasses import dataclass
from dune.fem.function import gridFunction
from dune.grid import structuredGrid
from math import pi, cos, exp, sqrt
from typing import Any

@dataclass
class Config1D:
    """
    starts parameters
    starts initial condition

    """
    #grid settings
    Nx: int = None
    Nv: int = None
    Nx_eval: int = None
    Nv_eval: int = None
    Mass: list = None
    Charge: list = None
    Ns: int =  1 # num of species
    S_name: str = "electrons"
    Nt_max: int  = None  # None => t_end/dt
    
    #spline settings
    order: int = 3 # spline order
    
    #sim settings
    dt: float = 0.1 # time step
    t_end: int = 20
    plot_freq: int = 5 # iterations between plots
    measure_freq: int = 1 # iterations between measurements

    k: float = 0.5 # wave nr
    eps: float = 1e-2 # perturbation amplitude
    
    v0: float = 3 # electron drift velocity

    grids: list = None
    


    it: int = None # simulation iteration tic
    time: float = 0
    time_array: list = None

    # Dune objects, definition at __post_init__
    gridView: Any = None
    expression_ini: Any = None
    fini: Any = None
        
    x_sampling_grid: np.ndarray = None
    v_sampling_grid: np.ndarray = None

    def __post_init__(self):
        self.Lx = 2*np.pi/self.k # spatial domain length
        self.Lv = 2*np.pi # velocity domain length

        if self.grids is None:
            self.grids = []
        #grid settings
        if self.Mass is None:
            self.Mass = [1]
        if self.Charge is None:
            self.Charge = [-1]
        if self.time_array is None:
            self.time_array = []
        if self.Nt_max is None:
            self.Nt_max = int(self.t_end/self.dt)+1
        if self.gridView is None:
            self.gridView = structuredGrid([0,-self.Lv], [self.Lx,self.Lv], [self.Nx-1, self.Nv-1])
        if self.expression_ini is None:
            self.expression_ini = lambda x: (1+self.eps * cos(self.k * x[0])) * x[1]**2 / (sqrt(2 * pi))* exp(-x[1]**2/2)
        
        # a dune.fem gridFunction
        self.fini = gridFunction(self.expression_ini, gridView=self.gridView, name='fini', order=self.order)
        self.dv = self.Lv/self.Nv_eval
        
        # Fourier wavenumbers
        self.kx = np.fft.fftshift((2*np.pi/self.Lx) * np.arange(-self.Nx_eval//2, self.Nx_eval//2))
        self.kx2 = self.kx**2
        self.kx2[0] = 1.0

        if self.x_sampling_grid is None:
            self.x_sampling_grid = np.linspace(0, self.Lx, num=self.Nx_eval)
        if self.v_sampling_grid is None:
            self.v_sampling_grid = np.linspace(-self.Lv, self.Lv, num=self.Nv_eval)


