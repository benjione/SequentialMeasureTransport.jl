abstract type Mapping{d, T} end

### methods interface for Mapping
pushforward(sampler::Mapping, u::PSDdata) = throw(NotImplementedError())
pullback(sampler::Mapping, x::PSDdata) = throw(NotImplementedError())
Jacobian(sampler::Mapping, x::PSDdata) = throw(NotImplementedError())
inverse_Jacobian(sampler::Mapping, u::PSDdata) = throw(NotImplementedError())
function pushforward(sampler::Mapping{d, T}, π::Function) where {d, T}
    π_pushed = let sampler=sampler, π=π
        (x) -> π(pullback(sampler, x)) * inverse_Jacobian(sampler, x)
    end
    return π_pushed
end
function pullback(sampler::Mapping{d, T}, π::Function) where {d, T}
    π_pulled = let sampler=sampler, π=π
        (u) -> π(pushforward(sampler, u)) * Jacobian(sampler, u)
    end
    return π_pulled
end

# include reference maps
include("reference_maps/reference_maps.jl")
using .ReferenceMaps

# include bridging densities
include("bridging/bridging_densities.jl")
using .BridgingDensities

"""
A Sampler is a mapping from a reference distribution to a target distribution,
while a mapping does not have any definition of a reference or target by itself.
"""
abstract type AbstractSampler{d, T, R} <: Mapping{d, T} end

"""
A Sampler of p(x, y) which can in addition conditionally sample
p(y|x), where y are the last dC variables.
"""
abstract type ConditionalSampler{d, T, R, dC} <: AbstractSampler{d, T, R} end

# a ConditionalSampler without conditional variables is a Sampler
const Sampler{d, T, R} = ConditionalSampler{d, T, R, 0}
is_conditional_sampler(::Type{<:ConditionalSampler}) = true
is_conditional_sampler(::Type{<:Sampler}) = false

Sampler(model::PSDModel) = @error "not implemented for this type of PSDModel"
ConditionalSampler(model::PSDModel, amount_cond_variable::Int) = @error "not implemented for this type of PSDModel"

function Base.show(io::IO, sampler::Sampler{d, T, R}) where {d, T, R}
    println(io, "Sampler{d=$d, T=$T, R=$R}")
end

import Serialization as ser
"""
For now, naive implementation of saving and loading samplers.
"""
function save_sampler(sampler::AbstractSampler, filename::String)
    ser.serialize(filename, sampler)
end
function load_sampler(filename::String)
    return ser.deserialize(filename)
end

## Interface for Sampler
Distributions.pdf(sampler::AbstractSampler, x::PSDdata) = throw(NotImplementedError())

## methods not necessarily implemented by concrete implementations of Sampler:
Base.rand(sampler::AbstractSampler{d, T}) where {d, T} = sample(sampler)
Base.rand(sampler::AbstractSampler{d, T}, amount::Int) where {d, T} = sample(sampler, amount)
Base.rand(sampler::AbstractSampler{d, T}, dims::Int...) where {d, T} = reshape(sample(sampler, prod(dims)), dims)
function sample(sampler::AbstractSampler{d, T}) where {d, T<:Number}
    return pushforward(sampler, sample_reference(sampler))
end
function sample(sampler::AbstractSampler{d, T}, amount::Int; threading=true) where {d, T}
    res = Vector{PSDdata{T}}(undef, amount)
    @_condusethreads threading for i=1:amount
        res[i] = sample(sampler)
    end
    return res
end

## methods for ConditionalSampler

# Helper functions already implemented
# marginal dimension
@inline _d_marg(
            _::ConditionalSampler{d, <:Any, <:Any, dC}
        ) where {d, dC} = d - dC

## interface for ConditionalSampler
"""
Distribution p(x) = ∫ p(x, y) d y
"""
marg_pdf(sampler::ConditionalSampler, x::PSDdata) = throw(NotImplementedError())
"""
pushforward of T(u) = y with u ∼ ρ and y ∼ p(y|x)
"""
cond_pushforward(sampler::ConditionalSampler, u::PSDdata, x::PSDdata) = throw(NotImplementedError())
cond_pullback(sampler::ConditionalSampler, y::PSDdata, x::PSDdata) = throw(NotImplementedError())

marg_pushforward(sampler::ConditionalSampler, u::PSDdata) = throw(NotImplementedError())
marg_pullback(sampler::ConditionalSampler, x::PSDdata) = throw(NotImplementedError())

# already implemented for ConditionalSampler with naive implementation
"""
PDF p(y|x)
"""
function cond_pdf(sampler::ConditionalSampler{d, T}, y::PSDdata{T}, x::PSDdata{T}) where {d, T}
    return Distributions.pdf(sampler, [x;y]) / marg_pdf(sampler, x)
end

function cond_pushforward(sampler::ConditionalSampler{d, T, <:Any, dC}, 
            u::PSDdata{T}, 
            x::PSDdata{T}
        ) where {d, T, dC}
    x = marg_pullback(sampler, x)
    xu = pushforward(sampler, [x;u])
    return xu[_d_marg(sampler)+1:d]
end

function cond_pullback(sra::ConditionalSampler{d, T, <:Any, dC}, 
                y::PSDdata{T}, 
                x::PSDdata{T}
            ) where {d, T<:Number, dC}
    yx = pullback(sra, [x; y])
    return yx[_d_marg(sampler)+1:end]
end

function cond_sample(sampler::ConditionalSampler{d, T}, 
                X::PSDDataVector{T};
                threading=true
            ) where {d, T<:Number}
    if threading == false
        return PSDdata{T}[cond_sample(sampler, x) for x in X]
    else
        res = Vector{PSDdata{T}}(undef, length(X))
        Threads.@threads for i=1:length(X)
            res[i] = cond_sample(sampler, X[i])
        end
        return res
    end
end
function cond_sample(sampler::ConditionalSampler{d, T, R, dC}, x::PSDdata{T}) where {d, T<:Number, R, dC}
    dx = _d_marg(sampler)
    return cond_pushforward(sampler, sample_reference(sampler)[dx+1:d], x)
end

## methods for Reference distribution
@inline _ref_pushforward(sampler::AbstractSampler{<:Any, T}, x::PSDdata{T}) where {T} = pushforward(sampler.R_map, x)
@inline _ref_pullback(sampler::AbstractSampler{<:Any, T}, u::PSDdata{T}) where {T} = pullback(sampler.R_map, u)
@inline _ref_Jacobian(sampler::AbstractSampler{<:Any, T}, x::PSDdata{T}) where {T} = Jacobian(sampler.R_map, x)
@inline _ref_inv_Jacobian(sampler::AbstractSampler{<:Any, T}, u::PSDdata{T}) where {T} = inverse_Jacobian(sampler.R_map, u)

@inline sample_reference(sampler::AbstractSampler{d, T, R}) where {d, T, R<:ReferenceMap} = ReferenceMaps.sample_reference(sampler.R_map)
@inline sample_reference(_::AbstractSampler{d, T, Nothing}) where {d, T} = rand(T, d)
@inline sample_reference(sampler::AbstractSampler{d, T, R}, n::Int) where {d, T, R<:ReferenceMap} = ReferenceMaps.sample_reference(sampler.R_map, n)
@inline sample_reference(_::AbstractSampler{d, T, Nothing}, n::Int) where {d, T} = rand(T, d, n)

@inline reference_pdf(sampler::AbstractSampler{d, T, R}, x) where {d, T, R<:ReferenceMap} = _ref_Jacobian(sampler, x)
@inline reference_pdf(_::AbstractSampler{d, T, Nothing}, x) where {d, T} = all(1.0 .> x .> 0) ? 1.0 : 0.0


include("PSDModelSampler.jl")
include("SubsetSampler.jl")
include("SelfReinforcedSampler.jl")