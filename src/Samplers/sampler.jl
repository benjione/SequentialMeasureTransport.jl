
abstract type Sampler{d, T} end

### methods implemented by concrete implmenetations:
Sampler(model::PSDModel) = @error "not implemented for this type of PSDModel"
pushforward_u(sampler::Sampler, u::PSDdata) = @error "not implemented for this type of Sampler"
pullback_x(sampler::Sampler, x::PSDdata) = @error "not implemented for this type of Sampler"
Distributions.pdf(sampler::Sampler, x::PSDdata) = @error "not implemented for this type of Sampler"

## methods not necessarily implemented by concrete implementations:
function sample(sampler::Sampler{d, T}) where {d, T<:Number}
    return pushforward_u(sampler, rand(T, d))
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

include("PSDModelSampler.jl")
include("SelfReinforcedSampler.jl")