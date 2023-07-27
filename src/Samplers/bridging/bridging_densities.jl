module BridgingDensities

abstract type BridgingDensity{d, T} end

using DifferentialEquations
using ..PSDModels: PSDDataVector, PSDdata

include("algebraic.jl")
include("diffusion.jl")

export BridgingDensity
export DiffusionBrigdingDensity, AlgebraicBridgingDensity
export evolve_samples

## Interface for bridging densities


## Sample based functions
evolve_samples(bridge::BridgingDensity, 
               X::PSDDataVector{T}, 
               t::T) where {T<:Number} = error("not implemented")




end # module BridgingDensities