import numpy as np
from dataclasses import dataclass

@dataclass
class Config1D:

    #grid settings
    Nsample: list = None
    Nx: int = None
    Nv: int = None
    Nmap: list = None
    Mass: list = None
    Charge: list = None
    Ns: int =  1 # num of species
    S_name: str = "electrons"
    Mr: int = 1 # Mass ratio for ions
    Nt_max: int  = 2000
    
    #spline settings
    order: int = 3 # spline order
    use_mex: bool = False 
    scheme: str = "lagrange-bary"
    
    #sim settings
    dt: float = 0.1 # time step
    dt_save: int = 5 # save after dt_same time
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

    #def f0(self, x, v):     #Physics version
    #    """
    #    clac. distribution
    #    """
    #    return (
    #        (1+self.eps * np.cos(self.k * x))
    #        / (2*np.sqrt(2 * np.pi))
    #        * (np.exp(-(v-self.v0)**2/2)+ np.exp(-(v+self.v0)**2/2))
    #    )

    def f0(self, x,v):     #Math version
        return (
            (1+self.eps * np.cos(self.k * x)) * v**2
            / (np.sqrt(2 * np.pi))
            * np.exp(-v**2/2)
        )
        
    def __post_init__(self):
        self.Lx = 2*np.pi/self.k # spatial domain length
        self.Lv = 2*np.pi # velocity domain length
        self.fini = [self.f0]

        if self.grids is None:
            self.grids = []
        #grid settings
        if self.Nsample is None:
            self.Nsample = [2**8, 2**8] # num of sample grid points
        if self.Nmap is None:
            self.Nmap = [2**6, 2**6] # num of saved map grid points 
        if self.Mass is None:
            self.Mass = [1]
        if self.Charge is None:
            self.Charge = [-1]
        if self.time_array is None:
            self.time_array = []
