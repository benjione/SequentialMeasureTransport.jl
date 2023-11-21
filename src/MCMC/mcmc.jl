module MCMC
"""
Implements MCMC based on the `AbstractMCMC` interface using the
sampler from `Sampler` in this package.
"""

using ..PSDModels
using ..PSDModels: ConditionalSampler
import AbstractMCMC
import MCMCChains
using Distributions
using Random
using LinearAlgebra

mutable struct MCMCSampler{T, S<:ConditionalSampler{<:Any, T}} <: AbstractMCMC.AbstractSampler
    current_state::Vector{T}
    sampler::S
end

struct MCMCModel{F} <: AbstractMCMC.AbstractModel
    model::F
    function MCMCModel(model::F) where {F}
        return new{F}(model)
    end
end
function (a::MCMCModel{F})(x::Vector{T}) where {F, T<:Number}
    return a.model(x)
end
function (a::MCMCModel{F})(x::Vector{T}, 
                sampler::ConditionalSampler{<:Any, T}
            ) where {F, T<:Number}
    return PSDModels.pullback(sampler, x->a.model(x))(x)
end

function proposal(sampler::MCMCSampler{T}, x::Vector{T}) where {T}
    return x + rand(MvNormal(zeros(length(sampler.current_state)), diagm(ones(length(sampler.current_state)))))
end

q(x::Vector{T}, y::Vector{T}) where {T<:Number} = pdf(MvNormal(zeros(length(x)), diagm(ones(length(x)))), x-y)

function AbstractMCMC.step(rng::Random.AbstractRNG,
        model::MCMCModel,
        samp::MCMCSampler{T};
        kwargs...) where {T <: Number}
    next_θ = PSDModels.sample(samp.sampler)
    return next_θ, next_θ
end
function AbstractMCMC.step(rng::Random.AbstractRNG,
        model::MCMCModel,
        samp::MCMCSampler{T},
        θ_prev::Vector{T};
        kwargs...) where {T <: Number}
    # θ_prev = samp.current_state
    # Generate a new proposal.
    # θ = propose(spl, model, θ_prev)
    θ = proposal(samp, θ_prev)

    # Calculate the acceptance probability.
    α = model(θ, samp.sampler) *  q(Θ, θ_prev) / (model(θ_prev, samp.sampler) * q(θ_prev, θ))

    # Decide whether to return the previous θ or the new one.
    next_Θ = if rand() < min(α, 1.0)
        θ
    else
        θ_prev
    end
    samp.current_state = next_Θ
    return next_Θ, next_Θ
end

# A basic chains constructor that works with the Transition struct we defined.
function AbstractMCMC.bundle_samples(
            ts::Vector{Vector{T}}, 
            ℓ::MCMCModel, 
            s::MCMCSampler{T}, 
            state, 
            chain_type::Type{Any};
            param_names=missing,
            kwargs...
        ) where {T<:Number}
    # Turn all the transitions into a vector-of-vectors.
    vals = reduce(hcat,ts)'

    # Check if we received any parameter names.
    if ismissing(param_names)
        param_names = String["Parameter $i" for i in 1:size(vals,2)]
    end

    # Bundle everything up and return a Chains struct.
    return MCMCChains.Chains(vals, param_names)
end


end # module MCMC