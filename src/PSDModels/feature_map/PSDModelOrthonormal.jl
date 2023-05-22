abstract type PSDModelOrthonormal{d, T, S} <: AbstractPSDModelFM{T} end

## special feature map models:
include("polynomial.jl")


"""
Φ(a::PSDModelFM, x::PSDdata{T}) where {T<:Number}

Returns the feature map of the PSD model at x.
"""
@inline function Φ(a::PSDModelOrthonormal{<:Any, T, S}, x::PSDdata{T}) where {T<:Number, S<:OMF{T}}
    return a.Φ(ξ(a.mapping, x)) * sqrt(1/x_deriv_prod(a.mapping, ξ(a.mapping, x)))
end


domain_interval_left(a::PSDModelOrthonormal{<:Any, <:Any, <:OMF}, k::Int) = -∞
domain_interval_right(a::PSDModelOrthonormal{<:Any, <:Any, <:OMF}, k::Int) = +∞

domain_interval_left(a::PSDModelOrthonormal, k::Int) = domain_interval(a, k)[1]
domain_interval_right(a::PSDModelOrthonormal, k::Int) = domain_interval(a, k)[2]
domain_interval_left(a::PSDModelOrthonormal{d}) where {d} = domain_interval_left.(Ref(a), collect(1:d))
domain_interval_right(a::PSDModelOrthonormal{d}) where {d} = domain_interval_right.(Ref(a), collect(1:d))

## general interface
_tensorizer(a::PSDModelOrthonormal) = throw(error("Not implemented!"))

## for greedy downward closed approximation
next_index_proposals(a::PSDModelOrthonormal) = next_index_proposals(_tensorizer(a))
create_proposal(a::PSDModelOrthonormal, index::Vector{Int}) = throw(error("Not implemented!"))