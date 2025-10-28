module ReferenceMaps

"""
    ReferenceMap{d, dC, T}

    Reference maps are diagonal maps used to work on domains of choice, while keeping the
    density estimation in [0, 1]. The reference map is defined as a map from the uniform
    distribution on [0, 1]^d to a domain of choice.

    R_♯ ρ = U
"""

import ..SequentialMeasureTransport as SMT
using ..SequentialMeasureTransport: PSDModelOrthonormal, domain_interval_left, domain_interval_right,
                   PSDdata, ConditionalMapping, pullback, pushforward, Jacobian, inverse_Jacobian

using SpecialFunctions: erf, erfcinv
using Distributions

# Maps to transform from U([0,1])^d to a domain of choice
# defined by R_# ρ = u where ρ is the reference distribution.
abstract type ReferenceMap{d, dC, T} <: ConditionalMapping{d, dC, T} end

include("scaling.jl")
include("gaussian.jl")
include("algebraic.jl")
include("composed_reference.jl")
include("super_heavy.jl")

export ReferenceMap
export ScalingReference, GaussianReference, AlgebraicReference
export normalized_gaussian_reference, normalized_algebraic_reference

"""
Attention!

Using any of the functions with dim(x) < d will take a marginal distribution and pdf.
"""


"""
Get a reference maps on the subspace of dimension, e.g. [3:4]
"""
function Base.getindex(m::ReferenceMap{d, dC, T}, I) where {d, dC, T}
    throw(error("Not implemented"))
end

Base.lastindex(m::ReferenceMap{d}) where {d} = d

### Interface for ReferenceMaps
@inline function Distributions.pdf(Rmap::ReferenceMap{<:Any, <:Any, T}, 
                        x::PSDdata{T}
                    ) where {T<:Number}
    Jacobian(Rmap, x)
end

function sample_reference(Rmap::ReferenceMap{d, <:Any, T}) where {d, T<:Number}
    pullback(Rmap, rand(T, d))
end

function sample_reference(Rmap::ReferenceMap{d, <:Any, T}, n::Int) where {d, T<:Number}
    map(z->pullback(Rmap, z), eachcol(rand(T, d, n)))
end

"""
    pushforward(mapping, x)

Pushes forward a vector of a reference distribution to the uniform distribution.
"""
function SMT.pushforward(
        mapping::ReferenceMap{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    throw(error("Not implemented"))
end

"""
    pullback(mapping, u)

Pulls back a vector of the uniform distribution to the reference distribution.
"""
function SMT.pullback(
        mapping::ReferenceMap{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    throw(error("Not implemented"))
end

"""
    Jacobian(mapping, x)

Computes the Jacobian of the mapping at the point u.
"""
function SMT.Jacobian(
        mapping::ReferenceMap{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    throw(error("Not implemented"))
end

"""
inverse_Jacobian(mapping, u)

Computes the inverse Jacobian of the mapping at the point x.
"""
function SMT.inverse_Jacobian(
        mapping::ReferenceMap{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    throw(error("Not implemented"))
end

function SMT.marginal_pushforward(
        mapping::ReferenceMap{d, dC, T}, 
        x::PSDdata{T}
    ) where{d, dC, T<:Number}
    throw(error("Not implemented"))
end

function SMT.marginal_pullback(
        mapping::ReferenceMap{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    throw(error("Not implemented"))
end

function SMT.marginal_Jacobian(
        mapping::ReferenceMap{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    throw(error("Not implemented"))
end

function SMT.marginal_inverse_Jacobian(
        mapping::ReferenceMap{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    throw(error("Not implemented"))
end

function SMT.conditional_log_Jacobian(sampler::ReferenceMap{<:Any, <:Any, T}, y::PSDdata{T}, x::PSDdata{T}) where {T}
    return SMT.log_Jacobian(sampler, [x; y]) - SMT.marginal_log_Jacobian(sampler, x)
end
function SMT.conditional_inverse_log_Jacobian(sampler::ReferenceMap{<:Any, <:Any, T}, y::PSDdata{T}, x::PSDdata{T}) where {T}
    return SMT.inverse_log_Jacobian(sampler, [x; y]) - SMT.marginal_inverse_log_Jacobian(sampler, x)
end


end # module ReferenceMaps