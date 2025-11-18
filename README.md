# PyNuFI 

dependencies: 
- numpy
- scipy
- matplotlib.pyplot
- ffmpeg

run simulation with ./simulate 

after simulating make .mp4 animation with ./animate


---

# notes to implement dune 

workflow:
- create a gridView
- make a space on your gridView (your spline space for each cell)
- make x=SpacialaCoordinate(space) (a ufl object)
- ufl expression to have distribution as np array by doing:
u_h.as_numpy[:]= space.interpolate(f(x))
this u_h will be your fs


--- 

# To-do

- initialize also if only Nx and Nv, if Nsamp or Nmap = none
