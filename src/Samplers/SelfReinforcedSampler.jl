
"""
A collection of self reinforced samplers, so that
maps of the type TODO
"""
struct SelfReinforcedSampler{d, T, R} <: Sampler{d, T, R}
    models::Vector{<:PSDModelOrthonormal{d, T}}
    samplers::Vector{<:Sampler{d, T}}
    R_map::R    # reference map from reference distribution to uniform
    function SelfReinforcedSampler(
        models::Vector{<:PSDModelOrthonormal{d, T}},
        samplers::Vector{<:Sampler{d, T}},
        R_map::R
    ) where {d, T<:Number, R<:ReferenceMap}
        @assert length(models) == length(samplers)
        new{d, T, R}(models, samplers, R_map)
    end
end

function Distributions.pdf(
        sar::SelfReinforcedSampler{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    pdf_func = pushforward_pdf_function(sar)
    return pdf_func(x)
end

function pullback_pdf_function(
        sra::SelfReinforcedSampler{d, T},
        pdf_tar::Function; # defined on [a, b]^d
        layers=nothing
    ) where {d, T<:Number}
    pdf_func = let pdf_tar=pdf_tar
        x->pdf_tar(x) # defined on [a, b]^d
    end
    _layers = layers === nothing ? (1:length(sra.models)) : layers
    # apply map R (domain transformation)
    pdf_func = let sra=sra, pdf_func=pdf_func
        u->begin
            pdf_func(_ref_pullback(sra, u)) * _ref_inv_Jacobian(sra, u)
        end
    end
    for (model, sampler) in zip(sra.models[_layers], sra.samplers[_layers])
        # apply map T_i
        pdf_func = let model = model, sra=sra, 
                        pdf_func=pdf_func, sampler=sampler
            u-> begin
                x = pushforward(sampler, u) # [0,1]->[a, b]
                return pdf_func(x) * (1/(model(x)))
            end
        end
    end
    # apply map R (domain transformation)
    pdf_func = let sra=sra, pdf_func=pdf_func
        x->begin
            pdf_func(_ref_pushforward(sra, x)) * _ref_Jacobian(sra, x) # [a, b]^d -> [0, 1]^d
        end
    end
    return pdf_func
end

function _pullback_jacobian_function(
        sra::SelfReinforcedSampler{d, T};
        layers=nothing
    ) where {d, T<:Number}
    jacob_func = x->1.0
    _layers = layers === nothing ? (1:length(sra.models)) : layers
    jacob_func = let jacob_func=jacob_func, sra=sra
        x->begin
            return jacob_func(_ref_pushforward(sra, x)) * _ref_Jacobian(sra, x)
        end
    end
    for j=_layers
        jacob_func = let jacob_func=jacob_func, sra=sra, j=j
            x->begin  
                x_for2 = pushforward(sra.samplers[j], x)
                return jacob_func(x_for2) * (1/(sra.models[j](x_for2)))
            end
        end
    end
    jacob_func = let jacob_func=jacob_func, sra=sra
        x->begin
            return jacob_func(_ref_pushforward(sra, x)) * _ref_Jacobian(sra, x)
        end
    end
    return jacob_func
end

function _broadcasted_pullback_pdf_function(
        sra::SelfReinforcedSampler{d, T},
        broad_pdf_tar::Function; # defined on [a, b]^d
        layers=nothing
    ) where {d, T<:Number}
    pdf_func = let broad_pdf_tar=broad_pdf_tar, jac_func=_pullback_jacobian_function(sra; layers)
        X->begin
            X_forwarded = pushforward.(Ref(sra), X)
            X_Jac = jac_func.(X)

            return broad_pdf_tar(X_forwarded) .* X_Jac
        end
    end
    return pdf_func
end

pushforward_pdf_function(sra::SelfReinforcedSampler; layers=nothing) = begin
    return pushforward_pdf_function(sra, x->reference_pdf(sra, x); layers=layers)
end
function pushforward_pdf_function(
        sra::SelfReinforcedSampler{d, T},
        pdf_ref::Function;
        layers=nothing
    ) where {d, T<:Number}
    pdf_func = let pdf_ref=pdf_ref
        x->begin
            pdf_ref(x)
        end
    end
    # from last to first
    _layers = layers === nothing ? (1:length(sra.models)) : layers
    pdf_func = let sra=sra, pdf_func=pdf_func
        u->begin
            pdf_func(_ref_pullback(sra, u)) * _ref_inv_Jacobian(sra, u)
        end
    end
    for (model, sampler) in zip(reverse(sra.models[_layers]), reverse(sra.samplers[_layers]))
        # apply map T_i
        pdf_func = let model=model, sra=sra,
                    pdf_func=pdf_func, sampler=sampler
            x-> begin
                pdf_func(pullback(sampler, x)) * model(x)
            end
        end
    end
    pdf_func = let sra=sra, pdf_func=pdf_func
        x->begin
            pdf_func(_ref_pushforward(sra, x)) * _ref_Jacobian(sra, x) # [a, b]^d -> [0, 1]^d
        end
    end
    return pdf_func
end

add_layer!(sra::SelfReinforcedSampler,
        pdf_tar::Function,
        fit_method!::Function;
        kwargs...
    ) = add_layer!(sra, pdf_tar, deepcopy(sra.models[1]), fit_method!; kwargs...)
function add_layer!(
        sra::SelfReinforcedSampler{d, T},
        pdf_tar::Function,
        model::PSDModelOrthonormal{d, T},
        fit_method!::Function;
        N_sample=1000,
        broadcasted_tar_pdf=false,
        kwargs...
    ) where {d, T<:Number}
    # sample from reference map
    X = eachcol(rand(T, d, N_sample))
    pdf_tar_pullbacked = if broadcasted_tar_pdf
        _broadcasted_pullback_pdf_function(sra, pdf_tar)        
    else 
        pullback_pdf_function(sra, pdf_tar)
    end
    pdf_tar_pullbacked_sample = if broadcasted_tar_pdf
        let π_tar=pdf_tar_pullbacked, sra=sra
            (x) -> π_tar(_ref_pullback.(Ref(sra), x)) .* _ref_inv_Jacobian.(Ref(sra), x)
        end
    else
        let π_tar=pdf_tar_pullbacked, sra=sra
            (x) -> π_tar(_ref_pullback(sra, x)) * _ref_inv_Jacobian(sra, x)
        end
    end
    Y = if broadcasted_tar_pdf
        pdf_tar_pullbacked_sample(X)
    else
        pdf_tar_pullbacked_sample.(X)
    end

    if any(isnan, Y)
        throw(error("NaN in target!"))
    end

    fit_method!(model, collect(X), Y)
    normalize!(model)
    push!(sra.models, model)
    push!(sra.samplers, Sampler(model))
    return nothing
end

function add_layer!(
        sra::SelfReinforcedSampler{d, T},
        model::PSDModelOrthonormal{d, T},
    ) where {d, T<:Number}
    normalize!(model)
    push!(sra.models, model)
    push!(sra.samplers, Sampler(model))
    return nothing
end


function SelfReinforcedSampler(
                pdf_tar::Function,
                model::PSDModelOrthonormal{d, T, S},
                amount_layers::Int,
                approx_method::Symbol,
                reference_map::ReferenceMap{d, T};
                ### for bridging densities
                relaxation_method::Symbol=:algebraic,
                N_sample=1000,
                max_blur=1.0,
                algebraic_base=2.0,
                N_relaxation=20, # number of MC for blurring
                langevin_time_grad=1.0,
                langevin_max_time=0.2,
                ### others
                broadcasted_tar_pdf=false,
                kwargs...) where {d, T<:Number, S}

    fit_method! = if approx_method == :Chi2
        if relaxation_method == :algebraic
            throw(error("Chi2 method does not support algebraic relaxation since the distribution "*
                         "has to be normalized, use Chi2U instead!"))
        end
        (m,x,y) -> Chi2_fit!(m, x, y; kwargs...)
    elseif approx_method == :Chi2U
        (m,x,y) -> Chi2U_fit!(m, x, y; kwargs...)
    else
        throw(error("Approx mehtod $(approx_method) not implemented!"))
    end

    L = domain_interval_left(model)
    R = domain_interval_right(model)
    @assert all(L .== 0.0)
    @assert all(R .== 1.0)
    if S<:OMF
        throw(error("Do not use OMF models for self reinforced sampler, use a Gaussian reference map instead!"))
    end

    ## Create bridging densities according to method
    π_tar = if relaxation_method == :algebraic
        if broadcasted_tar_pdf
            (X, β) -> pdf_tar(X).^β
        else
            (x,β) -> pdf_tar(x)^(β)
        end
    elseif relaxation_method == :blurring
        if broadcasted_tar_pdf
            (X, σ_i) -> let pdf_tar=pdf_tar, N_MC_blurring=N_relaxation, d=d
                begin
                    if σ_i == 0
                        return pdf_tar(X)
                    else
                        Res = zeros(T, length(X))
                        for i=1:N_MC_blurring
                            add_rand(x) = x .+ randn(T, d) .* σ_i
                            X_blurred = add_rand.(X)
                            Res .+= pdf_tar(X_blurred)
                        end
                        return (1/N_MC_blurring)*Res
                    end
                end
            end
        else
            (x, σ_i) -> let pdf_tar=pdf_tar, N_MC_blurring=N_relaxation, d=d
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
        end
    elseif relaxation_method == :langevin_diffusion
        if broadcasted_tar_pdf
            throw(error("Langevin diffusion with broadcasting not implemented"))
        else
            (x, t) -> let pdf_tar=pdf_tar, N_relaxation=N_relaxation, d=d
                begin
                    if t == 0
                        return pdf_tar(x)
                    else
                        X_rand = x .+ randn(T, d, N_relaxation) * (1 - exp(-2*t))^0.5
                        return (1/N_relaxation) * 1.0/(2 * π * (1 - exp(-2.0*t)))^(length(x)/2) * exp(t) * sum(
                            [pdf_tar(exp(t)*X_rand[:, k]) for k=1:N_relaxation]
                        )
                    end
                end
            end
        end
    elseif relaxation_method == :none
        (x,_) -> pdf_tar(x)
    else 
        throw(error("Not implemented"))
    end

    π_tar_samp = if broadcasted_tar_pdf
        let π_tar=π_tar, reference_map=reference_map
            (x, p) -> π_tar(pullback.(Ref(reference_map), x), p) .* inverse_Jacobian.(Ref(reference_map), x)
        end
    else
        let π_tar=π_tar, reference_map=reference_map
            (x, p) -> π_tar(pullback(reference_map, x), p) * inverse_Jacobian(reference_map, x)
        end
    end

    ## algebraic relaxation
    relax_param = if relaxation_method == :algebraic
        @assert algebraic_base > 1.0
        [algebraic_base^(-i) for i in reverse(0:amount_layers-1)]
    elseif relaxation_method == :blurring
        [[max_blur * (1/i^2) for i in 1:amount_layers-1]; [0.0]]
    elseif relaxation_method == :langevin_diffusion
        [[langevin_max_time^(langevin_time_grad*(i-1)+1) for i in 1:amount_layers-1]; [0.0]]
    elseif relaxation_method == :none
        [1.0 for i in 1:amount_layers]
    else
        throw(error("Not implemented!"))
    end

    # sample from reference map
    X = eachcol(rand(T, d, N_sample))
    # compute pdf
    Y = if broadcasted_tar_pdf
        π_tar_samp(X, relax_param[1]) 
    else
        π_tar_samp.(X, Ref(relax_param[1]))
    end

    if any(isnan, Y)
        throw(error("NaN in target!"))
    end

    fit_method!(model, X, Y)

    if any(isnan, model.B)
        throw(error("NaN in model! Model did not converge!"))
    end

    normalize!(model)
    samplers = [Sampler(model)]
    models = typeof(model)[model]

    sra = SelfReinforcedSampler(models, samplers, reference_map)

    for i in 2:amount_layers
        layer_method = let relax_param = relax_param
            x->π_tar(x, relax_param[i])
        end
        add_layer!(sra, layer_method, fit_method!; 
                N_sample=N_sample, 
                broadcasted_tar_pdf=broadcasted_tar_pdf,
                kwargs...)
    end
    
    return sra
end

function pushforward(
        sra::SelfReinforcedSampler{d, T}, 
        u::PSDdata{T};
        layers=nothing
    ) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.models)) : layers
    u = _ref_pushforward(sra, u)
    for j=reverse(_layers) # reverse order
        u = pushforward(sra.samplers[j], u)
    end
    u = _ref_pullback(sra, u)
    return u
end

function pullback(
        sra::SelfReinforcedSampler{d, T}, 
        x::PSDdata{T};
        layers=nothing
    ) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.models)) : layers
    x = _ref_pushforward(sra, x)
    for j=_layers
        x = pullback(sra.samplers[j], x)
    end
    x = _ref_pullback(sra, x)
    return x
end