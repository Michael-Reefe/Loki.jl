module Loki

using Reexport

include("cubefit.jl")
@reexport using .CubeFit

include("reduce.jl")
@reexport using .Reduce

#####################
#= WELCOME TO LOKI =#
#####################

# This file doesn't contain anything important, it just
# includes and reexports the other modules!

# Namespace:
# CubeData => cubedata.jl
# CubeFit => cubefit.jl
# Param => parameters.jl
# Util => utils.jl

# The functions and structs from CubeData and CubeFit that are exported are the main
# routines intended to be used to generate fits and models.  These modules have
# additional items that are not exported which are intended only for use internally.

# Functions and structs from Util and Param are similarly not exported 
# (i.e. must be prefixed by "Util" or "Param"), since again they are mostly intended
# only for use internally in the code.

end