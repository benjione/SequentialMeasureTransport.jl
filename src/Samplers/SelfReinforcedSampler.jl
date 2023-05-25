

struct SelfReinforcedSampler{d, T, S} <: Sampler{d, T}
    models::Vector{<:PSDModelOrthonormal{d, T, S}}
    samplers::Vector{<:Sampler{d, T}}
    mapping::S
    function SelfReinforcedSampler(
        models::Vector{<:PSDModelOrthonormal{d, T, S}},
        samplers::Vector{<:Sampler{d, T}}
    ) where {d, T<:Number, S}
        @assert length(models) == length(samplers)
        new{d, T, S}(models, samplers, models[1].mapping)
    end
    function SelfReinforcedSampler(
        models::Vector{<:PSDModelOrthonormal{d, T, S}},
        samplers::Vector{<:Sampler{d, T}},
        mapping::S
    ) where {d, T<:Number, S}
        @assert length(models) == length(samplers)
        new{d, T, S}(models, samplers, mapping)
    end
end


@inline function _domain_transform_jacobian(sar::SelfReinforcedSampler{d, T}) where {d, T<:Number}
    L_vec = domain_interval_right(sar.models[1]) - domain_interval_left(sar.models[1])
    V = prod(L_vec)
    return let V=V 
        _->1.0/V 
    end
end
@inline function _domain_transform_jacobian(sar::SelfReinforcedSampler{d, T, S}) where {d, T<:Number, S<:OMF}
    return let mapping=sar.mapping
        x->x_deriv_prod(mapping, 2*(x.-0.5)) * 2.0^(-d)
    end
end
@inline function _inverse_domain_transform_jacobian(sar::SelfReinforcedSampler{d, T}) where {d, T<:Number}
    L_vec = domain_interval_right(sar.models[1]) - domain_interval_left(sar.models[1])
    V = prod(L_vec)
    return let V=V 
        _->V 
    end
end
@inline function _inverse_domain_transform_jacobian(sar::SelfReinforcedSampler{d, T, S}) where {d, T<:Number, S<:OMF}
    return let mapping=sar.mapping 
        x->ξ_deriv_prod(mapping, x) * 2.0^d
    end
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
# domain transform from reference to target domain
function _domain_transform(
        sar::SelfReinforcedSampler{d, T, S}, 
        ξ::PSDdata{T}
    ) where {d, T<:Number, S<:OMF}
    ξ = ξ * 2.0 .- 1.0
    x(sar.mapping, ξ)
end
# inverse domain transform from target to reference domain
function _inverse_domain_transform(
        sar::SelfReinforcedSampler{d, T, S}, 
        x::PSDdata{T}
    ) where {d, T<:Number, S<:OMF}
    ξ_res = ξ(sar.mapping, x) # [-1, 1]^d
    return (ξ_res .+ 1.0) ./ 2.0 # [0, 1]^d
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
        pdf_tar::Function;
        layers=nothing
    ) where {d, T<:Number}
    pdf_func = let pdf_tar=pdf_tar
        x->pdf_tar(x)
    end
    _layers = layers === nothing ? (1:length(sar.models)) : layers
    for (model, sampler) in zip(sar.models[_layers], sar.samplers[_layers])
        pdf_func = let model = model, sar=sar, pdf_func=pdf_func, sampler=sampler
            x-> begin
                J = _inverse_domain_transform_jacobian(sar)
                u = _inverse_domain_transform(sar, x) # transform to reference domain
                x = pushforward_u(sampler, u)
                return (pdf_func(x) * (1/(model(x))))
            end
        end
    end
    return pdf_func
end

pushforward_pdf_function(sar::SelfReinforcedSampler; layers=nothing) = begin
    return pushforward_pdf_function(sar, x->1.0; layers=layers)
end
function pushforward_pdf_function(
        sar::SelfReinforcedSampler{d, T},
        pdf_ref::Function;
        layers=nothing
    ) where {d, T<:Number}
    pdf_func = let pdf_ref=pdf_ref, sar=sar
        J = _inverse_domain_transform_jacobian(sar)
        x->begin
            u = _inverse_domain_transform(sar, x)
            pdf_ref(u) * J(x)
        end
    end
    # from last to first
    _layers = layers === nothing ? (1:length(sar.models)) : layers
    for (model, sampler) in zip(reverse(sar.models[_layers]), reverse(sar.samplers[_layers]))
        pdf_func = let model=model, sar=sar,
                    pdf_func=pdf_func, sampler=sampler
            x-> begin
                u = pullback_x(sampler, x) # [a, b]^d -> [0, 1]^d
                J = _domain_transform_jacobian(sar)
                return (pdf_func(_domain_transform(sar, u)) * model(x) * J(u))                    
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

    if any(isnan, Y)
        throw(error("NaN in target!"))
    end

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
                relaxation_method::Symbol=:algebraic,
                N_sample=1000,
                max_blur=1.0,
                N_MC_blurring=20,
                kwargs...) where {d, T<:Number}

    fit_method! = if approx_method == :Chi2
        (m,x,y) -> Chi2_fit!(m, x, y; kwargs...)
    else
        throw(error("Approx mehtod $(approx_method) not implemented!"))
    end


    π_tar = if relaxation_method == :algebraic
        (x,β) -> pdf_tar(x)^(β)
    elseif relaxation_method == :blurring 
        (x, σ_i) -> let pdf_tar=pdf_tar, N_MC_blurring=N_MC_blurring, d=d
            begin
                if σ_i == 0
                    return pdf_tar(x)
                else
                    MC_samples = randn(T, d, N_MC_blurring) .* σ_i
                    return (1/N_MC_blurring)*sum(
                        [pdf_tar(x+MC_samples[:,k]) for k=1:N_MC_blurring]
                    )
                end
            end
        end
    elseif relaxation_method == :none
        (x,β) -> pdf_tar(x)
    else 
        throw(error("Not implemented"))
    end

    ## algebraic relaxation
    relax_param = if relaxation_method == :algebraic
        [2.0^(-i) for i in reverse(0:amount_layers-1)]
    elseif relaxation_method == :blurring
        [[max_blur * (1/i^2) for i in 1:amount_layers-1]; [0]]
    elseif relaxation_method == :none
        [1 for i in 1:amount_layers]
    else
        throw(error("Not implemented!"))
    end

    # sample from model
    X = x.(Ref(model), eachcol((rand(T, d, N_sample).-0.5) * 2))
    # compute pdf
    Y = π_tar.(X, Ref(relax_param[1]))

    if any(isnan, Y)
        throw(error("NaN in target!"))
    end

    fit_method!(model, X, Y)

    normalize!(model)
    samplers = [Sampler(model)]
    models = typeof(model)[model]

    sar = SelfReinforcedSampler(models, samplers)

    for i in 2:amount_layers
        layer_method = let relax_param = relax_param
            x->π_tar(x, relax_param[i])
        end
        add_layer!(sar, layer_method, approx_method; kwargs...)
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