
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
sample(sampler::Sampler{d, T}, amount::Int) where {d, T} = T[sample(sampler) for _=1:amount]

include("PSDModelSampler.jl")
include("SelfReinforcedSampler.jl")