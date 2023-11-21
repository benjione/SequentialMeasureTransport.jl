module ReferenceMaps

import ..PSDModels
using ..PSDModels: PSDModelOrthonormal, domain_interval_left, domain_interval_right,
                   PSDdata, Mapping, pullback, pushforward, Jacobian, inverse_Jacobian

using SpecialFunctions: erf, erfcinv
using Distributions

# Maps to transform from U([0,1])^d to a domain of choice
# defined by R_# ρ = u where ρ is the reference distribution.
abstract type ReferenceMap{d, T} <: Mapping{d, T} end

include("scaling.jl")
include("gaussian.jl")
include("algebraic.jl")

export ReferenceMap
export ScalingReference, GaussianReference, AlgebraicReference

"""
Attention!

Using any of the functions with dim(x) < d will take a marginal distribution and pdf.
"""

### Interface for ReferenceMaps
@inline function Distributions.pdf(Rmap::ReferenceMap{d, T}, 
                        x::PSDdata{T}
                    ) where {d, T<:Number}
    Jacobian(Rmap, x)
end

function sample_reference(Rmap::ReferenceMap{d, T}) where {d, T<:Number}
    pullback(Rmap, rand(T, d))
end

function sample_reference(Rmap::ReferenceMap{d, T}, n::Int) where {d, T<:Number}
    map(z->pullback(Rmap, z), eachcol(rand(T, d, n)))
end

"""
    pushforward(mapping, x)

Pushes forward a vector of a reference distribution to the uniform distribution.
"""
function PSDModels.pushforward(
        mapping::ReferenceMap{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    throw(error("Not implemented"))
end

"""
    pullback(mapping, u)

Pulls back a vector of the uniform distribution to the reference distribution.
"""
function PSDModels.pullback(
        mapping::ReferenceMap{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    throw(error("Not implemented"))
end

"""
    Jacobian(mapping, x)

Computes the Jacobian of the mapping at the point u.
"""
function PSDModels.Jacobian(
        mapping::ReferenceMap{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    throw(error("Not implemented"))
end

"""
inverse_Jacobian(mapping, u)

Computes the inverse Jacobian of the mapping at the point x.
"""
function PSDModels.inverse_Jacobian(
        mapping::ReferenceMap{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    throw(error("Not implemented"))
end


end # module ReferenceMaps