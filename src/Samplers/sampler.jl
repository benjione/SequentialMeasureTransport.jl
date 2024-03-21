
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

## Methods interface for ConditionalMapping

marg_pushforward(sampler::ConditionalMapping, u::PSDdata) = throw("Not Implemented")
marg_pullback(sampler::ConditionalMapping, x::PSDdata) = throw("Not Implemented")
marg_Jacobian(mapping::ConditionalMapping, u::PSDdata) = throw("Not Implemented")
marg_inverse_Jacobian(mapping::ConditionalMapping, x::PSDdata) = throw("Not Implemented")
function cond_Jacobian(sampler::ConditionalMapping{<:Any, <:Any,T}, y::PSDdata{T}, x::PSDdata{T}) where {T}
    x = marg_pullback(sampler, x)
    return Jacobian(sampler, [x; y]) / marg_Jacobian(sampler, x)
end
function cond_inverse_Jacobian(sampler::ConditionalMapping{<:Any, <:Any,T}, y::PSDdata{T}, x::PSDdata{T}) where {T}
    return inverse_Jacobian(sampler, [x; y]) / marg_inverse_Jacobian(sampler, x)
end

"""
Pullback of a conditional mapping.
x is from x ∼ π_x
"""
function cond_pushforward(sampler::ConditionalMapping{d,dC,T},
    u::PSDdata{T},
    x::PSDdata{T}
) where {d,T,dC}
    x = marg_pullback(sampler, x)
    xu = pushforward(sampler, [x; u])
    return xu[_d_marg(sampler)+1:d]
end
"""
Pullback of a conditional mapping.
x is from x ∼ π_x
"""
function cond_pullback(sra::ConditionalMapping{d,dC,T},
    y::PSDdata{T},
    x::PSDdata{T}
) where {d,T<:Number,dC}
    yx = pullback(sra, [x; y])
    return yx[_d_marg(sra)+1:end]
end
function marg_pullback(sampler::ConditionalMapping{d,dC,T}, π::Function) where {d,dC,T}
    π_pb = let sampler = sampler, π = π
        (x) -> begin
            π(marg_pushforward(sampler, x)) * marg_Jacobian(sampler, x)
        end
    end
    return π_pb
end
function marg_pushforward(sampler::ConditionalMapping{d,dC,T}, π::Function) where {d,dC,T}
    π_pf = let sampler = sampler, π = π
        (u) -> begin
            π(marg_pullback(sampler, u)) * marg_inverse_Jacobian(sampler, u)
        end
    end
    return π_pf
end
function cond_pushforward(sampler::ConditionalMapping{d,dC,T},
    π::Function,
    x::PSDdata{T}
) where {d,T<:Number,dC}
    π_pf = let sampler = sampler, π = π, x = x
        (y::PSDdata{T}) -> begin
            π(cond_pullback(sampler, y, x)) * cond_inverse_Jacobian(sampler, y, x)
        end
    end
    return π_pf
end

function cond_pullback(sampler::ConditionalMapping{d,dC,T},
    π::Function,
    x::PSDdata{T}
) where {d,T<:Number,dC}
    π_pb = let sampler = sampler, π = π, x = x
        (y::PSDdata{T}) -> begin
            π(cond_pushforward(sampler, y, x)) * cond_Jacobian(sampler, y, x)
        end
    end
    return π_pb
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
marg_pdf(sampler::AbstractCondSampler, x::PSDdata) = throw(NotImplementedError())
function marg_sample(sampler::AbstractCondSampler{d, dC, T}) where {d, dC, T<:Number}
    return marg_pushforward(sampler, sample_reference(sampler)[1:_d_marg(sampler)])
end
function marg_sample(sampler::AbstractCondSampler{<:Any,<:Any,T}, amount::Int; threading=true) where {T}
    res = Vector{Vector{T}}(undef, amount)
    @_condusethreads threading for i = 1:amount
        res[i] = marg_sample(sampler)
    end
    return res
end
# already implemented for ConditionalSampler with naive implementation
"""
PDF p(y|x) = p(x, y) / p(x)
"""
function cond_pdf(sampler::AbstractCondSampler{d,<:Any,T}, y::PSDdata{T}, x::PSDdata{T}) where {d,T}
    return Distributions.pdf(sampler, [x; y]) / marg_pdf(sampler, x)
end

function cond_sample(sampler::AbstractCondSampler{d,<:Any,T},
    X::PSDDataVector{T};
    threading=true
) where {d,T<:Number}
    if threading == false
        return PSDdata{T}[cond_sample(sampler, x) for x in X]
    else
        res = Vector{PSDdata{T}}(undef, length(X))
        Threads.@threads for i = 1:length(X)
            res[i] = cond_sample(sampler, X[i])
        end
        return res
    end
end
function cond_sample(sampler::AbstractCondSampler{d,dC,T,R}, x::PSDdata{T}) where {d,dC,T<:Number,R}
    dx = _d_marg(sampler)
    return cond_pushforward(sampler, sample_reference(sampler)[dx+1:d], x)
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

@inline marg_reference_pdf(sampler::AbstractCondSampler{d,<:Any,T,R}, x) where {d,T,R<:ReferenceMap} = marg_Jacobian(sampler.R1_map, x)
@inline marg_reference_pdf(_::AbstractCondSampler{d,<:Any,T,Nothing}, x) where {d,T} = all(1.0 .> x .> 0) ? 1.0 : 0.0

include("mappings/ProjectionMapping.jl")
include("mappings/MarginalMapping.jl")
include("samplers/PSDModelSampler.jl")
include("samplers/Sampler.jl")

include("algorithms.jl")