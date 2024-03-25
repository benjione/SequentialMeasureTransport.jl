
"""
A mapping is a function from a domain to a codomain.
"""
abstract type ConditionalMapping{d,dC,T} end
const Mapping{d,T} = ConditionalMapping{d,0,T}

"""
A mapping acting on a subset of the domain d of dimension dC.
"""
abstract type SubsetMapping{d,dC,T,dsub,dCsub} <: ConditionalMapping{d,dC,T} end

"""
A Sampler is a mapping from a reference distribution to a target distribution,
while a mapping does not have any definition of a reference or target by itself.
    T = R_1 ∘ Q_1 ∘ Q_2 ∘ ... ∘ Q_n ∘ R_2^{-1}
where R_1 is a map to a reference distribution and Q_i are maps on the hypercube [0,1]^d.
R_2 is a map from the domain of a distribution to the hypercube [0,1]^d.

Can be conditionally, so that p(x, y) is estimated and sampling of p(y | x) is possible.
"""
abstract type AbstractCondSampler{d,dC,T,R1,R2} <: ConditionalMapping{d,dC,T} end
const AbstractSampler{d,T,R1,R2} = AbstractCondSampler{d,0,T,R1,R2} # A sampler is also a Mapping

is_conditional(::Type{<:ConditionalMapping}) = true
is_conditional(::Type{<:Mapping}) = false

"""
Definition of interfaces.
"""

### methods interface for Mapping
pushforward(sampler::ConditionalMapping, u::PSDdata) = throw("Not Implemented")
pullback(sampler::ConditionalMapping, x::PSDdata) = throw("Not Implemented")
Jacobian(sampler::ConditionalMapping, x::PSDdata) = throw("Not Implemented")
inverse_Jacobian(sampler::ConditionalMapping, u::PSDdata) = throw("Not Implemented")
log_Jacobian(sampler::ConditionalMapping, x::PSDdata) = log(Jacobian(sampler, x))
inverse_log_Jacobian(sampler::ConditionalMapping, u::PSDdata) = log(inverse_Jacobian(sampler, u))
function pushforward(sampler::ConditionalMapping{d,<:Any,T}, π::Function) where {d,T<:Number}
    π_pushed = let sampler = sampler, π = π
        (x) -> π(pullback(sampler, x)) * inverse_Jacobian(sampler, x)
    end
    return π_pushed
end
function pullback(sampler::ConditionalMapping{d,<:Any,T}, π::Function) where {d,T<:Number}
    π_pulled = let sampler = sampler, π = π
        (u) -> π(pushforward(sampler, u)) * Jacobian(sampler, u)
    end
    return π_pulled
end
## log pushforward functions
function log_pushforward(sampler::ConditionalMapping{d,<:Any,T}, log_π::Function) where {d,T<:Number}
    π_pushed = let sampler = sampler, log_π = log_π
        (x) -> log_π(pullback(sampler, x)) + inverse_log_Jacobian(sampler, x)
    end
    return π_pushed
end
function log_pullback(sampler::ConditionalMapping{d,<:Any,T}, log_π::Function) where {d,T<:Number}
    π_pulled = let sampler = sampler, log_π = log_π
        (u) -> log_π(pushforward(sampler, u)) + log_Jacobian(sampler, u)
    end
    return π_pulled
end

## Methods interface for ConditionalMapping

marginal_pushforward(sampler::ConditionalMapping, u::PSDdata) = throw("Not Implemented")
marginal_pullback(sampler::ConditionalMapping, x::PSDdata) = throw("Not Implemented")
marginal_Jacobian(mapping::ConditionalMapping, u::PSDdata) = throw("Not Implemented")
marginal_inverse_Jacobian(mapping::ConditionalMapping, x::PSDdata) = throw("Not Implemented")
marginal_log_Jacobian(mapping::ConditionalMapping, u::PSDdata) = log(marginal_Jacobian(mapping, u))
marginal_inverse_log_Jacobian(mapping::ConditionalMapping, x::PSDdata) = log(marginal_inverse_Jacobian(mapping, x))
function conditional_Jacobian(sampler::ConditionalMapping{<:Any, <:Any,T}, y::PSDdata{T}, x::PSDdata{T}) where {T}
    x = marginal_pullback(sampler, x)
    return Jacobian(sampler, [x; y]) / marginal_Jacobian(sampler, x)
end
function conditional_inverse_Jacobian(sampler::ConditionalMapping{<:Any, <:Any,T}, y::PSDdata{T}, x::PSDdata{T}) where {T}
    return inverse_Jacobian(sampler, [x; y]) / marginal_inverse_Jacobian(sampler, x)
end
function conditional_log_Jacobian(sampler::ConditionalMapping{<:Any, <:Any,T}, y::PSDdata{T}, x::PSDdata{T}) where {T}
    x = marginal_pullback(sampler, x)
    return log_Jacobian(sampler, [x; y]) - marginal_log_Jacobian(sampler, x)
end
function conditional_inverse_log_Jacobian(sampler::ConditionalMapping{<:Any, <:Any,T}, y::PSDdata{T}, x::PSDdata{T}) where {T}
    return log_inverse_Jacobian(sampler, [x; y]) - marginal_inverse_log_Jacobian(sampler, x)
end

"""
Pullback of a conditional mapping.
x is from x ∼ π_x
"""
function conditional_pushforward(sampler::ConditionalMapping{d,dC,T},
    u::PSDdata{T},
    x::PSDdata{T}
) where {d,T,dC}
    x = marginal_pullback(sampler, x)
    xu = pushforward(sampler, [x; u])
    return xu[_d_marg(sampler)+1:d]
end
"""
Pullback of a conditional mapping.
x is from x ∼ π_x
"""
function conditional_pullback(sra::ConditionalMapping{d,dC,T},
    y::PSDdata{T},
    x::PSDdata{T}
) where {d,T<:Number,dC}
    yx = pullback(sra, [x; y])
    return yx[_d_marg(sra)+1:end]
end

##
## pushforward and pullback of functions
##

function marginal_pullback(sampler::ConditionalMapping{d,dC,T}, π::Function) where {d,dC,T}
    π_pb = let sampler = sampler, π = π
        (x) -> begin
            π(marginal_pushforward(sampler, x)) * marginal_Jacobian(sampler, x)
        end
    end
    return π_pb
end
function marginal_pushforward(sampler::ConditionalMapping{d,dC,T}, π::Function) where {d,dC,T}
    π_pf = let sampler = sampler, π = π
        (u) -> begin
            π(marginal_pullback(sampler, u)) * marginal_inverse_Jacobian(sampler, u)
        end
    end
    return π_pf
end

function marginal_log_pushforward(sampler::ConditionalMapping{d,dC,T}, log_π::Function) where {d,dC,T}
    log_π_pf = let sampler = sampler, log_π = log_π
        (u) -> begin
            log_π(marginal_pullback(sampler, u)) + marginal_inverse_log_Jacobian(sampler, u)
        end
    end
    return log_π_pf
end
function marginal_log_pullback(sampler::ConditionalMapping{d,dC,T}, log_π::Function) where {d,dC,T}
    log_π_pb = let sampler = sampler, log_π = log_π
        (x) -> begin
            log_π(marginal_pushforward(sampler, x)) + marginal_log_Jacobian(sampler, x)
        end
    end
    return log_π_pb
end

function conditional_pushforward(sampler::ConditionalMapping{d,dC,T},
    π::Function,
    x::PSDdata{T}
) where {d,T<:Number,dC}
    π_pf = let sampler = sampler, π = π, x = x
        (y::PSDdata{T}) -> begin
            π(conditional_pullback(sampler, y, x)) * conditional_inverse_Jacobian(sampler, y, x)
        end
    end
    return π_pf
end
function conditional_pullback(sampler::ConditionalMapping{d,dC,T},
    π::Function,
    x::PSDdata{T}
) where {d,T<:Number,dC}
    π_pb = let sampler = sampler, π = π, x = x
        (y::PSDdata{T}) -> begin
            π(conditional_pushforward(sampler, y, x)) * conditional_Jacobian(sampler, y, x)
        end
    end
    return π_pb
end
function conditional_log_pushforward(sampler::ConditionalMapping{d,dC,T},
    log_π::Function,
    x::PSDdata{T}
) where {d,T<:Number,dC}
    log_π_pf = let sampler = sampler, log_π = log_π, x = x
        (y::PSDdata{T}) -> begin
            log_π(conditional_pullback(sampler, y, x)) + conditional_inverse_log_Jacobian(sampler, y, x)
        end
    end
    return log_π_pf
end
function conditional_log_pullback(sampler::ConditionalMapping{d,dC,T},
    log_π::Function,
    x::PSDdata{T}
) where {d,T<:Number,dC}
    log_π_pb = let sampler = sampler, log_π = log_π, x = x
        (y::PSDdata{T}) -> begin
            log_π(conditional_pushforward(sampler, y, x)) + conditional_log_Jacobian(sampler, y, x)
        end
    end
    return log_π_pb
end

# include reference maps
include("reference_maps/reference_maps.jl")
using .ReferenceMaps

# include bridging densities
include("bridging/bridging_densities.jl")
using .BridgingDensities


Sampler(model::PSDModel) = @error "not implemented for this type of PSDModel"
ConditionalSampler(model::PSDModel, amount_cond_variable::Int) = @error "not implemented for this type of PSDModel"

function Base.show(io::IO, sampler::AbstractCondSampler{d,dC,T,R1,R2}) where {d,dC,T,R1,R2}
    println(io, "Sampler{d=$d, dC=$dC, T=$T, R1=$R1, R2=$R2}")
end

import Serialization as ser
"""
For now, naive implementation of saving and loading samplers.
"""
function save_sampler(sampler::AbstractCondSampler, filename::String)
    ser.serialize(filename, sampler)
end
function load_sampler(filename::String)
    return ser.deserialize(filename)
end

## Interface for Sampler
Distributions.pdf(sampler::AbstractCondSampler, x::PSDdata) = throw(NotImplementedError())
Distributions.logpdf(sampler::AbstractCondSampler, x::PSDdata) = throw(NotImplementedError())

## methods not necessarily implemented by concrete implementations of Sampler:
Base.rand(sampler::AbstractCondSampler) = sample(sampler)
Base.rand(sampler::AbstractCondSampler{d,<:Any,T}, amount::Int) where {d,T} = sample(sampler, amount)
Base.rand(sampler::AbstractCondSampler{d,<:Any,T}, dims::Int...) where {d,T} = reshape(sample(sampler, prod(dims)), dims)
function sample(sampler::AbstractCondSampler{d,<:Any,T}) where {d,T<:Number}
    return pushforward(sampler, sample_reference(sampler))
end
function sample(sampler::AbstractCondSampler{d,<:Any,T}, amount::Int; threading=true) where {d,T}
    res = Vector{Vector{T}}(undef, amount)
    @_condusethreads threading for i = 1:amount
        res[i] = sample(sampler)
    end
    return res
end

## methods for ConditionalSampler

# Helper functions already implemented
# marginal dimension
@inline _d_marg(
    _::ConditionalMapping{d,dC}
) where {d,dC} = d - dC

## interface for ConditionalSampler
"""
Distribution p(x) = ∫ p(x, y) d y
"""
marginal_pdf(sampler::AbstractCondSampler, x::PSDdata) = throw(NotImplementedError())
marginal_logpdf(sampler::AbstractCondSampler, x::PSDdata) = throw(NotImplementedError())
function marginal_sample(sampler::AbstractCondSampler{d, dC, T}) where {d, dC, T<:Number}
    return marginal_pushforward(sampler, sample_reference(sampler)[1:_d_marg(sampler)])
end
function marginal_sample(sampler::AbstractCondSampler{<:Any,<:Any,T}, amount::Int; threading=true) where {T}
    res = Vector{Vector{T}}(undef, amount)
    @_condusethreads threading for i = 1:amount
        res[i] = marginal_sample(sampler)
    end
    return res
end
# already implemented for ConditionalSampler with naive implementation
"""
PDF p(y|x) = p(x, y) / p(x)
"""
function conditional_pdf(sampler::AbstractCondSampler{d,<:Any,T}, y::PSDdata{T}, x::PSDdata{T}) where {d,T}
    return Distributions.pdf(sampler, [x; y]) / marginal_pdf(sampler, x)
end
function conditional_logpdf(sampler::AbstractCondSampler{d,<:Any,T}, y::PSDdata{T}, x::PSDdata{T}) where {d,T}
    return Distributions.logpdf(sampler, [x; y]) - Distributions.marg_logpdf(sampler, x)
end

function conditional_sample(sampler::AbstractCondSampler{d,<:Any,T},
    X::PSDDataVector{T};
    threading=true
) where {d,T<:Number}
    if threading == false
        return PSDdata{T}[conditional_sample(sampler, x) for x in X]
    else
        res = Vector{PSDdata{T}}(undef, length(X))
        Threads.@threads for i = 1:length(X)
            res[i] = conditional_sample(sampler, X[i])
        end
        return res
    end
end
function conditional_sample(sampler::AbstractCondSampler{d,dC,T,R}, x::PSDdata{T}) where {d,dC,T<:Number,R}
    dx = _d_marg(sampler)
    return conditional_pushforward(sampler, sample_reference(sampler)[dx+1:d], x)
end

## methods for Reference distribution
@inline _ref_pushforward(sampler::AbstractCondSampler{<:Any,<:Any,T}, x::PSDdata{T}) where {T} = pushforward(sampler.R1_map, x)
@inline _ref_pullback(sampler::AbstractCondSampler{<:Any,<:Any,T}, u::PSDdata{T}) where {T} = pullback(sampler.R1_map, u)
@inline _ref_Jacobian(sampler::AbstractCondSampler{<:Any,<:Any,T}, x::PSDdata{T}) where {T} = Jacobian(sampler.R1_map, x)
@inline _ref_inv_Jacobian(sampler::AbstractCondSampler{<:Any,<:Any,T}, u::PSDdata{T}) where {T} = inverse_Jacobian(sampler.R1_map, u)

@inline sample_reference(sampler::AbstractCondSampler{d,<:Any,T,R}) where {d,T,R<:ReferenceMap} = ReferenceMaps.sample_reference(sampler.R1_map)
@inline sample_reference(_::AbstractCondSampler{d,<:Any,T,Nothing}) where {d,T} = rand(T, d)
@inline sample_reference(sampler::AbstractCondSampler{d,<:Any,T,R}, n::Int) where {d,T,R<:ReferenceMap} = ReferenceMaps.sample_reference(sampler.R1_map, n)
@inline sample_reference(_::AbstractCondSampler{d,<:Any,T,Nothing}, n::Int) where {d,T} = rand(T, d, n)

@inline reference_pdf(sampler::AbstractCondSampler{d,<:Any,T,R}, x) where {d,T,R<:ReferenceMap} = _ref_Jacobian(sampler, x)
@inline reference_pdf(_::AbstractCondSampler{d,<:Any,T,Nothing}, x) where {d,T} = all(1.0 .> x .> 0) ? 1.0 : 0.0

@inline marginal_reference_pdf(sampler::AbstractCondSampler{d,<:Any,T,R}, x) where {d,T,R<:ReferenceMap} = marginal_Jacobian(sampler.R1_map, x)
@inline marginal_reference_pdf(_::AbstractCondSampler{d,<:Any,T,Nothing}, x) where {d,T} = all(1.0 .> x .> 0) ? 1.0 : 0.0

include("mappings/ProjectionMapping.jl")
include("mappings/MarginalMapping.jl")
include("samplers/PSDModelSampler.jl")
include("samplers/Sampler.jl")

include("algorithms.jl")