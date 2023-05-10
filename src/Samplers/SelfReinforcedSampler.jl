

struct SelfReinforcedSampler{d, T} <: Sampler{d, T}
    models::Vector{<:PSDModelOrthonormal{d, T}}
    samplers::Vector{<:Sampler{d, T}}
    function SelfReinforcedSampler(
        models::Vector{<:PSDModelOrthonormal{d, T}},
        samplers::Vector{<:Sampler{d, T}}
    ) where {d, T<:Number}
        @assert length(models) == length(samplers)
        new{d, T}(models, samplers)
    end
end


@inline function _domain_transform_constant(sar::SelfReinforcedSampler{d, T}) where {d, T<:Number}
    L_vec = domain_interval_right(sar.models[1]) - domain_interval_left(sar.models[1])
    V = prod(L_vec)
    return V
end

# domain transform from reference to target domain
function _domain_transform(
        sar::SelfReinforcedSampler{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    L = domain_interval_left(sar.models[1])
    R = domain_interval_right(sar.models[1])
    return x .* (R .- L) .+ L
end
# inverse domain transform from target to reference domain
function _inverse_domain_transform(
        sar::SelfReinforcedSampler{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    L = domain_interval_left(sar.models[1])
    R = domain_interval_right(sar.models[1])
    return (x .- L) ./ (R .- L)
end


function Distributions.pdf(
        sar::SelfReinforcedSampler{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    pdf_func = pushforward_pdf_function(sar)
    return pdf_func(x)
end

function pullback_pdf_function(
        sar::SelfReinforcedSampler{d, T},
        pdf_tar::Function
    ) where {d, T<:Number}
    pdf_func = let pdf_tar=pdf_tar
        x->pdf_tar(x)
    end
    for (model, sampler) in zip(sar.models, sar.samplers)
        pdf_func = let model = model, sar=sar, pdf_func=pdf_func, sampler=sampler
            x-> begin
                c = _domain_transform_constant(sar)
                u = _inverse_domain_transform(sar, x) # transform to reference domain
                x = pushforward_u(sampler, u)
                return (pdf_func(x) * (1/(model(x) * c)))
            end
        end
    end
    return pdf_func
end

pushforward_pdf_function(sar::SelfReinforcedSampler) = begin
    c = _domain_transform_constant(sar)
    return pushforward_pdf_function(sar, x->1/c)
end
function pushforward_pdf_function(
        sar::SelfReinforcedSampler{d, T},
        pdf_ref::Function
    ) where {d, T<:Number}
    pdf_func = let pdf_ref=pdf_ref, sar=sar
        x->pdf_ref(_inverse_domain_transform(sar, x))
    end
    # from last to first
    for (model, sampler) in zip(reverse(sar.models), reverse(sar.samplers))
        pdf_func = let model=model, sar=sar,
                    pdf_func=pdf_func, sampler=sampler
            x-> begin
                c = _domain_transform_constant(sar)
                u = pullback_x(sampler, x)
                u = _domain_transform(sar, u) # transform to target domain
                return (pdf_func(u) * model(x) * c)                    
            end
        end
    end
    return pdf_func
end

add_layer!(sar::SelfReinforcedSampler,
        pdf_tar::Function,
        approx_method::Symbol;
        kwargs...
    ) = add_layer!(sar, pdf_tar, deepcopy(sar.models[1]), approx_method; kwargs...)
function add_layer!(
        sar::SelfReinforcedSampler{d, T},
        pdf_tar::Function,
        model::PSDModelOrthonormal{d, T},
        approx_method::Symbol;
        kwargs...
    ) where {d, T<:Number}
    
    fit_method! = if approx_method == :Chi2
        (m,x,y) -> Chi2_fit!(m, x, y; kwargs...)
    else
        throw(error("Approx mehtod $(approx_method) not implemented!"))
    end
    
    X = _domain_transform.(Ref(sar), eachcol(rand(T, d, 1000)))
    pdf_tar_pullbacked = pullback_pdf_function(sar, pdf_tar)
    Y = pdf_tar_pullbacked.(X)

    fit_method!(model, collect(X), Y)
    normalize!(model)
    push!(sar.models, model)
    push!(sar.samplers, Sampler(model))
    return nothing
end

function add_layer!(
        sar::SelfReinforcedSampler{d, T},
        model::PSDModelOrthonormal{d, T},
    ) where {d, T<:Number}
    normalize!(model)
    push!(sar.models, model)
    push!(sar.samplers, Sampler(model))
    return nothing
end


function SelfReinforcedSampler(
                pdf_tar::Function,
                model::PSDModelOrthonormal{d, T},
                amount_layers::Int,
                approx_method::Symbol;
                kwargs...) where {d, T<:Number}
    
    L = domain_interval_left(model)
    R = domain_interval_right(model)

    _inverse_domain_transform(x) = (x .- L) ./ (R .- L)

    fit_method! = if approx_method == :Chi2
        (m,x,y) -> Chi2_fit!(m, x, y; kwargs...)
    else
        throw(error("Approx mehtod $(approx_method) not implemented!"))
    end

    ## algebraic relaxation
    β = [2.0^(-i) for i in reverse(0:amount_layers-1)]

    # sample from model
    X = rand(T, d, 1000) .* (R .- L) .+ L
    # compute pdf
    Y = pdf_tar.(eachcol(X)).^(β[1])

    fit_method!(model, collect(eachcol(X)), Y)

    normalize!(model)
    samplers = [Sampler(model)]
    models = typeof(model)[model]

    sar = SelfReinforcedSampler(models, samplers)

    for i in 2:amount_layers
        add_layer!(sar, x->pdf_tar(x)^β[i], approx_method; kwargs...)
    end
    
    return sar
end

function pushforward_u(
        sra::SelfReinforcedSampler{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    x = similar(u)
    for sampler in sra.samplers
        x = pushforward_u(sampler, u)
        u = _inverse_domain_transform(sra, x)
    end
    return x
end

function pullback_x(
        sra::SelfReinforcedSampler{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    u = similar(x)
    for sampler in sra.samplers
        u = pullback_x(sampler, x)
        x = _domain_transform(sra, u)
    end
    return u
end