# PyNuFI 

dependencies: 
- numpy
- scipy
- matplotlib.pyplot
- ffmpeg
- DUNE-grid
- DUNE-common
- DUNE-fem

run simulation with `./simulate`

after simulating make .mp4 animation with `./animate`

--- 

# To-do

> [!NOTE]
> funciton eval problem is because grid isnt periodic => when I evaluate my f on the spline it isnt periodic
> I need to use a dune.grid.yaspGrid with a dune.grid.cartesianDomain
> when defining my cartesianDomain i need to set periodic=True
> look at line 70 from [dune-example](https://github.com/dune-project/dune-grid/blob/f3ba252b/python/dune/grid/tutorial/example.py)
