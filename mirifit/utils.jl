module Utils

export C_MS, Σ, extend, rest_frame

using NaNStatistics

# Useful constants
const C_MS = 299792458

# Useful functions

# Sum along specific dimensions, ignoring nans, and dropping those dimensions
Σ(array, dims) = dropdims(nansum(array, dims=dims), dims=dims)

# Extend a 1D array into other dimensions
extend(array1d, shape) = repeat(reshape(array1d, (1,1,length(array1d))), outer=[shape...,1])

# Convert wavelength to rest-frame
rest_frame(λ, z) = @. λ / (1 + z)

end