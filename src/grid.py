from dataclasses import dataclass
from dune.grid import structuredGrid
from typing import Any
# ------- Grid Class ------- #

@dataclass
class Grid:
    # spacing
    dv: float = 0.0

    # domain lengths
    Lx: float = 0.0
    Lv: float = 0.0

    # number of points
    Nx: int = 0
    Nv: int = 0

    method: str = "spline"

    # Dune grid, a structuredGrid object
    gridView: Any = None

# ------- Grid funcs ------- #
def make_periodic_grid(Lx, Lv, Nx, Nv):
    # 1D grids
    dv = Lv/Nv
    
    # Dune gridView
    gridView = structuredGrid([0,-Lv], [Lx,Lv], [Nx-1,Nv-1])
    
   
    grid = Grid(
        dv=dv,
        Lx=Lx,
        Lv=Lv,
        Nx=Nx,
        Nv=Nv,
        method="spline",
        gridView=gridView,
    )

    return grid
