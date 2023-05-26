

# Maps to transform from U([0,1])^d to a domain of choice
# defined by R_# ρ = u where ρ is the reference distribution.
abstract type ReferenceMap{d, T} end

include("scaling.jl")
include("gaussian.jl")


## Convenient Constructors
# IdentityReference(d, T) = ScalingReference{d}(zeros(T, d), ones(T, d))

### Interface for ReferenceMaps

function sample_reference(map::ReferenceMap{d, T}) where {d, T<:Number}
    pushforward(map, rand(T, d))
end

function sample_reference(map::ReferenceMap{d, T}, n::Int) where {d, T<:Number}
    map(z->pushforward(map, z), rand(T, d, n))
end

"""
    pushforward(mapping, x)

Pushes forward a vector of a reference distribution to the uniform distribution.
"""
function pushforward(
        mapping::ReferenceMap{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    throw(error("Not implemented"))
end

"""
    pullback(mapping, u)

Pulls back a vector of the uniform distribution to the reference distribution.
"""
function pullback(
        mapping::ReferenceMap{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    throw(error("Not implemented"))
end

"""
    Jacobian(mapping, x)

Computes the Jacobian of the mapping at the point u.
"""
function Jacobian(
        mapping::ReferenceMap{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    throw(error("Not implemented"))
end

"""
inverse_Jacobian(mapping, u)

Computes the inverse Jacobian of the mapping at the point x.
"""
function inverse_Jacobian(
        mapping::ReferenceMap{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    throw(error("Not implemented"))
end