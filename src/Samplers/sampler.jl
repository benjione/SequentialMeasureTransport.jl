abstract type Mapping{d, T} end

### methods interface for Mapping
pushforward(sampler::Mapping, u::PSDdata) = throw(NotImplementedError())
pullback(sampler::Mapping, x::PSDdata) = throw(NotImplementedError())

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
abstract type Sampler{d, T, R} <: Mapping{d, T} end

"""
A Sampler of p(x, y) which can in addition conditionally sample
p(y|x), where y are the last dC variables.
"""
abstract type ConditionalSampler{d, T, R, dC} <: Sampler{d, T, R} end


Sampler(model::PSDModel) = @error "not implemented for this type of PSDModel"
ConditionalSampler(model::PSDModel, amount_cond_variable::Int) = @error "not implemented for this type of PSDModel"

function Base.show(io::IO, sampler::Sampler{d, T, R}) where {d, T, R}
    println(io, "Sampler{d=$d, T=$T, R=$R}")
end

## Interface for Sampler
Distributions.pdf(sampler::Sampler, x::PSDdata) = throw(NotImplementedError())
pushforward(sampler::Sampler, u::PSDdata) = throw(NotImplementedError())
pullback(sampler::Sampler, x::PSDdata) = throw(NotImplementedError())

## methods not necessarily implemented by concrete implementations of Sampler:
Base.rand(sampler::Sampler{d, T}) where {d, T} = sample(sampler)
Base.rand(sampler::Sampler{d, T}, amount::Int) where {d, T} = sample(sampler, amount)
Base.rand(sampler::Sampler{d, T}, dims::Int...) where {d, T} = reshape(sample(sampler, prod(dims)), dims)
function sample(sampler::Sampler{d, T}) where {d, T<:Number}
    return pushforward(sampler, sample_reference(sampler))
end
function sample(sampler::Sampler{d, T}, amount::Int; threading=false) where {d, T}
    if threading==false
        return PSDdata{T}[sample(sampler) for _=1:amount]
    else
        res = Vector{PSDdata{T}}(undef, amount)
        Threads.@threads for i=1:amount
            res[i] = sample(sampler)
        end
        return res
    end
end

## methods for ConditionalSampler
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

# already implemented for ConditionalSampler
"""
PDF p(y|x)
"""
function cond_pdf(sampler::ConditionalSampler{d, T}, y::PSDdata{T}, x::PSDdata{T}) where {d, T}
    return Distributions.pdf(sampler, [x;y]) / marg_pdf(sampler, x)
end

## methods for Reference distribution
@inline _ref_pushforward(sampler::Sampler{<:Any, T}, x::PSDdata{T}) where {T} = pushforward(sampler.R_map, x)
@inline _ref_pullback(sampler::Sampler{<:Any, T}, u::PSDdata{T}) where {T} = pullback(sampler.R_map, u)
@inline _ref_Jacobian(sampler::Sampler{<:Any, T}, x::PSDdata{T}) where {T} = Jacobian(sampler.R_map, x)
@inline _ref_inv_Jacobian(sampler::Sampler{<:Any, T}, u::PSDdata{T}) where {T} = inverse_Jacobian(sampler.R_map, u)

@inline sample_reference(sampler::Sampler{d, T, R}) where {d, T, R<:ReferenceMap} = ReferenceMaps.sample_reference(sampler.R_map)
@inline sample_reference(_::Sampler{d, T, Nothing}) where {d, T} = rand(T, d)
@inline sample_reference(sampler::Sampler{d, T, R}, n::Int) where {d, T, R<:ReferenceMap} = ReferenceMaps.sample_reference(sampler.R_map, n)
@inline sample_reference(_::Sampler{d, T, Nothing}, n::Int) where {d, T} = rand(T, d, n)

@inline reference_pdf(sampler::Sampler{d, T, R}, x) where {d, T, R<:ReferenceMap} = _ref_Jacobian(sampler, x)
@inline reference_pdf(_::Sampler{d, T, Nothing}, x) where {d, T} = all(1.0 .> x .> 0) ? 1.0 : 0.0


include("PSDModelSampler.jl")
include("SubsetSampler.jl")
include("SelfReinforcedSampler.jl")