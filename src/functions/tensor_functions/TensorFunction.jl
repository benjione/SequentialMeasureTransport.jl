
include("tensorizers/Tensorizers.jl")

abstract type TensorFunction{d, T, S<:Tensorizer{d}} <: Function end


dimensions(::TensorFunction{d}) where {d} = d


@inline σ(p::TensorFunction{<:Any, <:Any, S}, i) where {S<:Tensorizer} = σ(p.ten, i)
@inline σ_inv(p::TensorFunction{<:Any, <:Any, S}, i) where {S<:Tensorizer} = σ_inv(p.ten, i)


include("TensorPolynomial.jl")
include("MappedTensorFunction.jl")