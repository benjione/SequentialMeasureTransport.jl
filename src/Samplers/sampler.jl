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

Sampler(model::PSDModel) = @error "not implemented for this type of PSDModel"
Distributions.pdf(sampler::Sampler, x::PSDdata) = @error "not implemented for this type of Sampler"


## methods not necessarily implemented by concrete implementations:
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