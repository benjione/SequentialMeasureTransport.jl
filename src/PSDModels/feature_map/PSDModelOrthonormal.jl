abstract type PSDModelOrthonormal{d, T} <: AbstractPSDModelFM{T} end

## special feature map models:
include("polynomial.jl")



domain_interval_left(a::PSDModelOrthonormal, k::Int) = domain_interval(a, k)[1]
domain_interval_right(a::PSDModelOrthonormal, k::Int) = domain_interval(a, k)[2]
domain_interval_left(a::PSDModelOrthonormal{d}) where {d} = domain_interval_left.(Ref(a), collect(1:d))
domain_interval_right(a::PSDModelOrthonormal{d}) where {d} = domain_interval_right.(Ref(a), collect(1:d))

## general interface
_tensorizer(a::PSDModelOrthonormal) = throw(error("Not implemented!"))

## for greedy downward closed approximation
next_index_proposals(a::PSDModelOrthonormal) = next_index_proposals(_tensorizer(a))
create_proposal(a::PSDModelOrthonormal, index::Vector{Int}) = throw(error("Not implemented!"))