

"""
The most generic implementation of CondSampler and Sampler.
"""
struct CondSampler{d, dC, T, R1, R2} <: AbstractCondSampler{d, dC, T, R1, R2}
    samplers::Vector{<:ConditionalMapping{d, dC, T}}   # defined on [0, 1]^d
    R1_map::R1   # reference map from reference distribution to uniform on [0, 1]^d
    R2_map::R2   # distribution from domain of pi to [0, 1]^d
    function CondSampler(
        samplers::Vector{<:ConditionalMapping{d, dC, T}},
        R1_map::R1,
        R2_map::R2
    ) where {d, T<:Number, R1<:Union{<:ReferenceMap, Nothing}, 
            R2<:Union{<:ReferenceMap, Nothing}, dC}
        new{d, dC, T, R1, R2}(samplers, R1_map, R2_map)
    end
    function CondSampler(
        samplers::Vector{<:ConditionalMapping},
        R_map::R
    ) where {R<:Union{<:ReferenceMap, Nothing}}
        CondSampler(samplers, R_map, R_map)
    end
    function CondSampler{d, dC, T}(
        R1_map::R1,
        R2_map::R2
    ) where {d, dC, T<:Number, 
            R1<:Union{<:ReferenceMap, Nothing}, 
            R2<:Union{<:ReferenceMap, Nothing}}
        CondSampler(ConditionalMapping{d, dC, T}[], R1_map, R2_map)
    end
end

function Sampler(
            mappings::Vector{<:Mapping{d, T}},
            R1_map::Union{<:ReferenceMap, Nothing},
            R2_map::Union{<:ReferenceMap, Nothing};
        ) where {d, T<:Number}
    return CondSampler(mappings, R1_map, R2_map)
end

function Sampler(
            mappings::Vector{<:Mapping{d, T}},
            R_map::Union{<:ReferenceMap, Nothing}
        ) where {d, T<:Number}
    return CondSampler(mappings, R_map)
end

## Pretty printing
function Base.show(io::IO, sra::CondSampler{d, <:Any, T}) where {d, T<:Number}
    println(io, "SelfReinforcedSampler{d=$d, T=$T}")
    println(io, "  samplers:")
    for (i, sampler) in enumerate(sra.samplers)
        if i>3
            println(io, "...")
            break
        end
        println(io, "    $i: $sampler")
    end
    println(io, "  reference map: $(sra.R1_map), $(sra.R2_map)")
end

## Overwrite pdf function from Distributions
function Distributions.pdf(
        sar::CondSampler{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    pdf_func = pushforward_pdf_function(sar)
    return pdf_func(x)
end

function pullback_pdf_function(
        sra::CondSampler{d, <:Any, T},
        pdf_tar::Function; # defined on [a, b]^d
        layers=nothing
    ) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    # apply map R (domain transformation)
    pdf_func = pushforward(sra.R2_map, pdf_tar)
    for sampler in sra.samplers[_layers]
        # apply map T_i
        pdf_func = pullback(sampler, pdf_func)
    end
    return pullback(sra.R1_map, pdf_func)
end


function _broadcasted_pullback_pdf_function(
        sra::CondSampler{d, <:Any, T},
        broad_pdf_tar::Function; # defined on [a, b]^d
        layers=nothing
    ) where {d, T<:Number}
    pdf_func = let broad_pdf_tar=broad_pdf_tar, jac_func=pullback_pdf_function(sra, x->one(T); layers)
        X->begin
            X_forwarded = pushforward.(Ref(sra), X)
            X_Jac = jac_func.(X)

            return broad_pdf_tar(X_forwarded) .* X_Jac
        end
    end
    return pdf_func
end

pushforward_pdf_function(sra::CondSampler; layers=nothing) = begin
    return pushforward(sra, x->reference_pdf(sra, x); layers=layers)
end
function pushforward(
        sra::CondSampler{d, <:Any, T},
        pdf_ref::Function;
        layers=nothing
    ) where {d, T<:Number}
    # from last to first
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    pdf_func = pushforward(sra.R2_map, pdf_ref)
    for sampler in reverse(sra.samplers[_layers])
        # apply map T_i
        pdf_func = pushforward(sampler, pdf_func)
    end
    return pullback(sra.R1_map, pdf_func)
end


marg_pushforward_pdf_function(sra::CondSampler; layers=nothing) = begin
    return marg_pushforward(sra, x->reference_pdf(sra, x); layers=layers)
end
function marg_pushforward(
        sra::CondSampler{d, <:Any, T},
        pdf_ref::Function;
        layers=nothing
    ) where {d, T<:Number}
    # from last to first
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    pdf_func = pushforward(sra.R2_map, pdf_ref)
    for sampler in reverse(sra.samplers[_layers])
        # apply map T_i
        pdf_func = marg_pushforward(sampler, pdf_func)
    end
    pdf_func = pullback(sra.R1_map, pdf_func)
    return pdf_func
end


function add_layer!(
        sra::CondSampler{d, dC, T},
        sampler::ConditionalMapping{d, dC, T},
    ) where {d, T<:Number, dC}
    push!(sra.samplers, sampler)
    return nothing
end


## Interface of Sampler

function pushforward(
        sra::CondSampler{d, <:Any, T}, 
        u::PSDdata{T};
        layers=nothing
    ) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    u = pushforward(sra.R2_map, u)
    for j=reverse(_layers) # reverse order
        u = pushforward(sra.samplers[j], u)
    end
    u = pullback(sra.R1_map, u)
    return u::Vector{T}
end

function pullback(
        sra::CondSampler{d, <:Any, T}, 
        x::PSDdata{T};
        layers=nothing
    ) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    x = pushforward(sra.R1_map, x)
    for j=_layers
        x = pullback(sra.samplers[j], x)
    end
    x = pullback(sra.R2_map, x)
    return x::Vector{T}
end

"""
TODO: can be done prettier
"""
function inverse_Jacobian(sra::CondSampler{d, <:Any, T},
            x::PSDdata{T}) where {d, T<:Number}
    pushforward(sra, x->one(T))(x)
end
@inline Jacobian(sra::CondSampler{d, <:Any, T},
            x::PSDdata{T}) where {d, T<:Number} = 1/inverse_Jacobian(sra, pushforward(sra, x))

## Interface of ConditionalSampler

function marg_pdf(sra::CondSampler{d, dC, T, R1, R2}, x::PSDdata{T}) where {d, dC, T<:Number, R1, R2}
    @assert length(x) == _d_marg(sra)
    pdf_func = marg_pushforward_pdf_function(sra)
    return pdf_func(x)
end

function marg_pushforward(sra::CondSampler{d, <:Any, T}, u::PSDdata{T};
                        layers=nothing) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    u = pushforward(sra.R2_map, u)
    for j=reverse(_layers) # reverse order
        u = marg_pushforward(sra.samplers[j], u)
    end
    u = pullback(sra.R1_map, u)
    return u
end

function marg_pullback(sra::CondSampler{d, <:Any, T}, x::PSDdata{T};
                        layers=nothing) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    x = pushforward(sra.R1_map, x)
    for j=_layers
        x = marg_pullback(sra.samplers[j], x)
    end
    x = pullback(sra.R2_map, x)
    return x
end

