
include("tensorizers/Tensorizers.jl")

abstract type TensorFunction{d, T, S<:Tensorizer{d}} <: Function end

include("TensorPolynomial.jl")