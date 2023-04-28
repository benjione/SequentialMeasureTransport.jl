module PSDModels

using LinearAlgebra, SparseArrays
using KernelFunctions: Kernel, kernelmatrix
using DomainSets
using FastGaussQuadrature: gausslegendre
using ApproxFun
using Combinatorics: multiexponents
import ForwardDiff as FD
import Base

include("utils.jl")
include("optimization.jl")

export PSDModel
## export optimization
export fit!, minimize!, IRLS!

## export differentiation and integration
export gradient, integrate
export integral
export normalize!, normalize_orth_measure!
export marginalize_orth_measure, marginalize

## export arithmetic
export mul!

# for working with 1D and nD data
const PSDdata{T} = Union{T, Vector{T}} where {T<:Number}
const PSDDataVector{T} = Union{Vector{T}, Vector{Vector{T}}} where {T<:Number}

abstract type PSDModel{T} end
abstract type TraceModel{T} end

include("functions/functions.jl")
include("PSDModels/models.jl")
include("TraceModels/models.jl")

# tailored statistics for PSD models
include("statistics.jl")
using .Statistics

end # module PositiveSemidefiniteModels
