

"""
The most generic implementation of CondSampler and Sampler.
"""
struct CondSampler{d, dC, T, R1, R2} <: AbstractCondSampler{d, dC, T, R1, R2}
    samplers::Vector{<:ConditionalMapping{d, dC, T}}   # defined on [0, 1]^d
    R1_map::R1   # reference map from reference distribution to uniform on [0, 1]^d
    R2_map::R2   # distribution from domain of pi to [0, 1]^d
    function CondSampler(
        samplers::Vector{<:ConditionalMapping{d, dC, T}},
        R1_map::Union{<:ReferenceMap{d, dC, T}, Nothing},
        R2_map::Union{<:ReferenceMap{d, dC, T}, Nothing}
    ) where {d, T<:Number, dC}
        new{d, dC, T, typeof(R1_map), typeof(R2_map)}(samplers, R1_map, R2_map)
    end
    function CondSampler(
        samplers::Vector{<:ConditionalMapping},
        R_map::R
    ) where {R<:Union{<:ReferenceMap, Nothing}}
        CondSampler(samplers, R_map, R_map)
    end
    function CondSampler{d, dC, T}(
        R1_map::Union{<:ReferenceMap, Nothing},
        R2_map::Union{<:ReferenceMap, Nothing}
    ) where {d, dC, T<:Number}
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

function concatenate(sra::CondSampler{d, dC, T, R1, R2}, srb::CondSampler{d, dC, T, R1, R2}) where {d, dC, T<:Number, R1, R2}
    samplers = vcat(sra.samplers, srb.samplers)
    return CondSampler(samplers, sra.R1_map, sra.R2_map)
end

## Pretty printing
function Base.show(io::IO, sra::CondSampler{d, <:Any, T}) where {d, T<:Number}
    println(io, "SelfReinforcedSampler{d=$d, T=$T}")
    println(io, "$(length(sra.samplers)) layers")
    println(io, "  reference map: $(sra.R1_map), $(sra.R2_map)")
end

function _broadcasted_pullback_pdf_function(
        sra::CondSampler{d, <:Any, T},
        broad_pdf_tar::Function; # defined on [a, b]^d
        layers=nothing
    ) where {d, T<:Number}
    pdf_func = let broad_pdf_tar=broad_pdf_tar, jac_func=pullback(sra, x->one(T); layers)
        X->begin
            X_forwarded = pushforward.(Ref(sra), X)
            X_Jac = jac_func.(X)

            return broad_pdf_tar(X_forwarded) .* X_Jac
        end
    end
    return pdf_func
end

#### Macros for pushforward and pullback functions

macro pushforward_function(prefix)
    quote
        function $(esc(Symbol(prefix, "pushforward")))(
            sra::CondSampler{d, <:Any, T},
            pdf_ref::Function;
            layers=nothing
        ) where {d, T<:Number}
            _layers = layers === nothing ? (1:length(sra.samplers)) : layers
            pdf_func = $(esc(Symbol(prefix, "pushforward")))(sra.R2_map, pdf_ref)
            for j=reverse(_layers) # reverse order
                pdf_func = $(esc(Symbol(prefix, "pushforward")))(sra.samplers[j], pdf_func)
            end
            return $(esc(Symbol(prefix, "pullback")))(sra.R1_map, pdf_func)
        end
    end
end

macro pullback_function(prefix)
    quote
        function $(esc(Symbol(prefix, "pullback")))(
            sra::CondSampler{d, <:Any, T},
            π::Function;
            layers=nothing
        ) where {d, T<:Number}
            _layers = layers === nothing ? (1:length(sra.samplers)) : layers
            pdf_func = $(esc(Symbol(prefix, "pushforward")))(sra.R1_map, π)
            for j=_layers
                pdf_func = $(esc(Symbol(prefix, "pullback")))(sra.samplers[j], pdf_func)
            end
            return $(esc(Symbol(prefix, "pullback")))(sra.R2_map, pdf_func)
        end
    end
end

macro pushforward_sample(prefix)
    quote
        function $(esc(Symbol(prefix, "pushforward")))(
            sra::CondSampler{d, <:Any, T}, 
            u::PSDdata{T};
            layers=nothing
        ) where {d, T<:Number}
            _layers = layers === nothing ? (1:length(sra.samplers)) : layers
            u = $(esc(Symbol(prefix, "pushforward")))(sra.R2_map, u)
            for j=reverse(_layers) # reverse order
                u = $(esc(Symbol(prefix, "pushforward")))(sra.samplers[j], u)
            end
            u = $(esc(Symbol(prefix, "pullback")))(sra.R1_map, u)
            return u::Vector{T}
        end     
    end
end

macro pullback_sample(prefix)
    quote
        function $(esc(Symbol(prefix, "pullback")))(
            sra::CondSampler{d, <:Any, T}, 
            u::PSDdata{T};
            layers=nothing
        ) where {d, T<:Number}
            _layers = layers === nothing ? (1:length(sra.samplers)) : layers
            u = $(esc(Symbol(prefix, "pushforward")))(sra.R1_map, u)
            for j=_layers
                u = $(esc(Symbol(prefix, "pullback")))(sra.samplers[j], u)
            end
            u = $(esc(Symbol(prefix, "pullback")))(sra.R2_map, u)
            return u::Vector{T}
        end     
    end
end

###########################################
# Function pushforward and pullback

@pushforward_function ""
@pullback_function ""

@pushforward_function "log_"
@pullback_function "log_"

@pushforward_function "marginal_"
@pullback_function "marginal_"

@pushforward_function "marginal_log_"
@pullback_function "marginal_log_"

###########################################
# Sample pushforward and pullback

@pushforward_sample ""
@pullback_sample ""

@pushforward_sample "marginal_"
@pullback_sample "marginal_"


function add_layer!(
        sra::CondSampler{d, dC, T},
        sampler::ConditionalMapping{d, dC, T},
    ) where {d, T<:Number, dC}
    push!(sra.samplers, sampler)
    return nothing
end


"""
Jacobian defintions
"""

function inverse_Jacobian(sra::CondSampler{d, <:Any, T},
            x::PSDdata{T}) where {d, T<:Number}
    pushforward(sra, x->one(T))(x)
end
@inline Jacobian(sra::CondSampler{d, <:Any, T},
            x::PSDdata{T}) where {d, T<:Number} = pullback(sra, x->one(T))(x)

function marginal_inverse_Jacobian(sra::CondSampler{d, dC, T, R1, R2}, x::PSDdata{T}) where {d, dC, T<:Number, R1, R2}
    return marginal_pushforward(sra, x->one(T))(x)
end

function marginal_Jacobian(sra::CondSampler{d, dC, T, R1, R2}, x::PSDdata{T}) where {d, dC, T<:Number, R1, R2}
    return marginal_pullback(sra, x->one(T))(x)
end


"""
PDF definitions
"""
## Overwrite pdf function from Distributions
function Distributions.pdf(
        sar::CondSampler{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    pdf_func = pushforward(sar, x->reference_pdf(sar, x))
    return pdf_func(x)
end

function marginal_pdf(sra::CondSampler{d, dC, T, R1, R2}, x::PSDdata{T}) where {d, dC, T<:Number, R1, R2}
    @assert length(x) == _d_marg(sra)
    # reference Jacobian flexible for dimension
    pdf_func = marginal_pushforward(sra, x->marginal_reference_pdf(sra, x))
    return pdf_func(x)
end
