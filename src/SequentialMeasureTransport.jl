module SequentialMeasureTransport

using LinearAlgebra, SparseArrays
using KernelFunctions: Kernel, kernelmatrix
using DomainSets
using FastGaussQuadrature: gausslegendre
using ApproxFun
using Combinatorics: multiexponents
using Transducers
import ForwardDiff as FD
import StatsBase

## overwrite functions in this module for compatibility
import Base
import Distributions
import Random

include("config.jl")
include("utils.jl")

export PSDModel
## export optimization
export fit!, minimize!, IRLS!

## export differentiation and integration
export gradient
export integral
export normalize!, normalize_orth_measure!
export marginalize_orth_measure, marginalize

## export arithmetic
export mul!

# export sampler
export Sampler, ConditionalSampler
export SelfReinforcedSampler

# for working with 1D and nD data
const PSDdata{T} = AbstractVector{T} where {T<:Number}
const PSDDataVector{T} = AbstractVector{<:AbstractVector{T}} where {T<:Number}

# function models
abstract type PSDModel{T} end
abstract type TraceModel{T} end

"""
    abstract type ConditionalMapping{d,dC,T}

Abstract type representing a conditional mapping for variables ``(x, y) \\in \\mathcal{R}^d`` and
``y \\in \\mathcal{R}^{dC}``. Mapping can be either used to map ``(z_x, z_y)`` to ``(x, y)``, or
given ``x`` to map ``z_y`` to ``y``. Such maps are triangular of the form:
``
    \\mathccal{T}(z_x, z_y) = (T_x(z_x), T_y(z_x, z_y)) = (x, y)
``

- `d`: The dimension of the input space.
- `dC`: The dimension of the conditioning space.
- `T`: Number type used, e.g. Float64.

"""
abstract type ConditionalMapping{d,dC,T} end

"""
    const Mapping{d,T} = ConditionalMapping{d,0,T}

A `Mapping` type that is an alias for `ConditionalMapping` with `dC = 0`, hence, non conditional.
This type represents a mapping between two sets of measures.

"""
const Mapping{d,T} = ConditionalMapping{d,0,T}

include("functions/functions.jl")
include("PSDModels/models.jl")
include("TraceModels/models.jl")

# adaptive sampling from target Distributions
include("extra/adaptive_sampling/stopping_rule_MC_sampling.jl")

# Optimization methods on models
include("optimization/optimization.jl")

# Samplers for PSD models
include("Samplers/sampler.jl")

# tailored statistics for PSD models and samplers
include("statistics.jl")
using .Statistics

# methods to create Optimal Transport plans and more
include("OptimalTransport.jl")
using .OptimalTransport

end # module PositiveSemidefiniteModels
