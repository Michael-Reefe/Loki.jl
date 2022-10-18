module Loki

using Reexport

include("cubefit.jl")
@reexport using .CubeFit

# Namespace:
# CubeData => cubedata.jl
# CubeFit => cubefit.jl
# Param => parameters.jl
# Util => utils.jl

end