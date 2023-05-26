
# include reference maps
include("reference_maps/reference_maps.jl")

abstract type Sampler{d, T, R} end

### methods implemented by concrete implmenetations:
Sampler(model::PSDModel) = @error "not implemented for this type of PSDModel"
pushforward(sampler::Sampler, u::PSDdata) = @error "not implemented for this type of Sampler"
pullback(sampler::Sampler, x::PSDdata) = @error "not implemented for this type of Sampler"
Distributions.pdf(sampler::Sampler, x::PSDdata) = @error "not implemented for this type of Sampler"

## methods not necessarily implemented by concrete implementations:
function sample(sampler::Sampler{d, T}) where {d, T<:Number}
    return pushforward(sampler, rand(T, d))
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

@inline sample_reference(sampler::Sampler) = sample_reference(sampler.R_map)

@inline reference_pdf(sampler::Sampler, x) = _ref_Jacobian(sampler, x)


include("PSDModelSampler.jl")
include("SelfReinforcedSampler.jl")