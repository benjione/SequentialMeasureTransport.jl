module PSDModels

using LinearAlgebra, SparseArrays
using KernelFunctions: Kernel, kernelmatrix
using ProximalOperators: IndPSD, prox, prox!
using DomainSets
using FastGaussQuadrature: gausslegendre
import ForwardDiff as FD
import ProximalAlgorithms
import Base

include("utils.jl")
include("optimization.jl")

export PSDModel
export fit!, minimize!

## export differentiation and integration
export gradient, integrate
export integral
export marginalize_orth_measure

## export arithmetic
export mul!

# for working with 1D and nD data
const PSDdata{T} = Union{T, Vector{T}} where {T<:Number}
const PSDDataVector{T} = Union{Vector{T}, Vector{Vector{T}}} where {T<:Number}

abstract type PSDModel{T} end
abstract type TraceModel{T} end

include("PSDModels/models.jl")

end # module PositiveSemidefiniteModels
