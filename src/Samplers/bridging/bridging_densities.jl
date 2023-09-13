module BridgingDensities

abstract type BridgingDensity{d, T} end

using DifferentialEquations
using ..PSDModels: PSDDataVector, PSDdata

include("algebraic.jl")
include("diffusion.jl")
include("alpha_geodesic.jl")

export BridgingDensity
export DiffusionBrigdingDensity, AlgebraicBridgingDensity, AlphaGeodesicBridgingDensity
export evolve_samples, evaluate_bridge

## Interface for bridging densities
evaluate_bridge(bridge::BridgingDensity{d, T}, 
                x::PSDdata{T}, 
                k::Int) where {d, T<:Number} = error("not implemented")
function (bridge::BridgingDensity{d, T})(x::PSDdata{T}, k::Int) where {d, T<:Number}
    evaluate_bridge(bridge, x, k)
end

# for broadcasting
evaluate_bridge(bridge::BridgingDensity{d, T}, 
                X::PSDDataVector{T}, 
                k::Int) where {d, T<:Number} = error("not implemented")
function (bridge::BridgingDensity{d, T})(X::PSDDataVector{T}, k::Int) where {d, T<:Number}
    evaluate_bridge(bridge, X, k)
end

## Sample based functions
evolve_samples(bridge::BridgingDensity{<:Any, T}, 
               X::PSDDataVector{T}, 
               t::T) where {T<:Number} = error("not implemented")




end # module BridgingDensities