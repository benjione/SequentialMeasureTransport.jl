

struct SelfReinforcedSampler{d, T, R} <: Sampler{d, T, R}
    models::Vector{<:PSDModelOrthonormal{d, T}}
    samplers::Vector{<:Sampler{d, T}}
    R_map::R    # reference map from reference distribution to uniform
    function SelfReinforcedSampler(
        models::Vector{<:PSDModelOrthonormal{d, T}},
        samplers::Vector{<:Sampler{d, T}},
        R_map::R
    ) where {d, T<:Number, R}
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
        sar::SelfReinforcedSampler{d, T},
        pdf_tar::Function; # defined on [a, b]^d
        layers=nothing
    ) where {d, T<:Number}
    pdf_func = let pdf_tar=pdf_tar
        x->pdf_tar(x) # defined on [a, b]^d
    end
    _layers = layers === nothing ? (1:length(sar.models)) : layers
    for (model, sampler) in zip(sar.models[_layers], sar.samplers[_layers])
        # apply map T_i
        pdf_func = let model = model, sar=sar, 
                        pdf_func=pdf_func, sampler=sampler
            u-> begin
                x = pushforward(sampler, u) # [0,1]->[a, b]
                return pdf_func(x) * (1/(model(x)))
            end
        end
        # apply map R (domain transformation)
        pdf_func = let sar=sar, pdf_func=pdf_func
            x->begin
                pdf_func(_ref_pushforward(sar, x)) * _ref_Jacobian(sar, x) # [a, b]^d -> [0, 1]^d
            end
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
    for j=_layers
        jacob_func = let jacob_func=jacob_func, sra=sra, j=j
            x->begin
                x_for = _ref_pushforward(sra, x)    
                x_for2 = pushforward(sra.samplers[j], x_for)
                return jacob_func(x_for2) * _ref_Jacobian(sra, x) * (1/(sra.models[j](x_for2)))
            end
        end
    end
    return jacob_func
end

function _broadcasted_pullback_pdf_function(
        sar::SelfReinforcedSampler{d, T},
        broad_pdf_tar::Function; # defined on [a, b]^d
        layers=nothing
    ) where {d, T<:Number}
    pdf_func = let broad_pdf_tar=broad_pdf_tar, jac_func=_pullback_jacobian_function(sar; layers)
        X->begin
            X_forwarded = pushforward.(Ref(sar), X)
            X_Jac = jac_func.(X)

            return broad_pdf_tar(X_forwarded) .* X_Jac
        end
    end
    return pdf_func
end

pushforward_pdf_function(sar::SelfReinforcedSampler; layers=nothing) = begin
    return pushforward_pdf_function(sar, x->reference_pdf(sar, x); layers=layers)
end
function pushforward_pdf_function(
        sar::SelfReinforcedSampler{d, T},
        pdf_ref::Function;
        layers=nothing
    ) where {d, T<:Number}
    pdf_func = let pdf_ref=pdf_ref
        x->begin
            pdf_ref(x)
        end
    end
    # from last to first
    _layers = layers === nothing ? (1:length(sar.models)) : layers
    for (model, sampler) in zip(reverse(sar.models[_layers]), reverse(sar.samplers[_layers]))
        # apply map R (domain transformation)
        pdf_func = let sar=sar, pdf_func=pdf_func
            u->begin
                pdf_func(_ref_pullback(sar, u)) * _ref_inv_Jacobian(sar, u)
            end
        end
        # apply map T_i
        pdf_func = let model=model, sar=sar,
                    pdf_func=pdf_func, sampler=sampler
            x-> begin
                pdf_func(pullback(sampler, x)) * model(x)
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
        N_sample=1000,
        broadcasted_tar_pdf=false,
        kwargs...
    ) where {d, T<:Number}
    
    fit_method! = if approx_method == :Chi2
        (m,x,y) -> Chi2_fit!(m, x, y; kwargs...)
    else
        throw(error("Approx mehtod $(approx_method) not implemented!"))
    end
    
    X = sample_reference(sar, N_sample)
    pdf_tar_pullbacked = if broadcasted_tar_pdf
        _broadcasted_pullback_pdf_function(sar, pdf_tar)        
    else 
        pullback_pdf_function(sar, pdf_tar)
    end
    Y = if broadcasted_tar_pdf
        pdf_tar_pullbacked(X)
    else
        pdf_tar_pullbacked.(X)
    end

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
                model::PSDModelOrthonormal{d, T, S},
                amount_layers::Int,
                approx_method::Symbol;
                relaxation_method::Symbol=:algebraic,
                N_sample=1000,
                max_blur=1.0,
                algebraic_base=2.0,
                N_MC_blurring=20,
                reference_map=nothing,
                broadcasted_tar_pdf=false,
                kwargs...) where {d, T<:Number, S}

    fit_method! = if approx_method == :Chi2
        (m,x,y) -> Chi2_fit!(m, x, y; kwargs...)
    else
        throw(error("Approx mehtod $(approx_method) not implemented!"))
    end

    reference_map = if reference_map !== nothing
        reference_map
    elseif S<:OMF
        GaussianReference{d, T}()
    else 
        ScalingReference(model)
    end


    π_tar = if relaxation_method == :algebraic
        if broadcasted_tar_pdf
            (X, β) -> pdf_tar(X).^β
        else
            (x,β) -> pdf_tar(x)^(β)
        end
    elseif relaxation_method == :blurring
        if broadcasted_tar_pdf
            (X, σ_i) -> let pdf_tar=pdf_tar, N_MC_blurring=N_MC_blurring, d=d
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
        end
    elseif relaxation_method == :none
        (x,_) -> pdf_tar(x)
    else 
        throw(error("Not implemented"))
    end

    ## algebraic relaxation
    relax_param = if relaxation_method == :algebraic
        @assert algebraic_base > 1.0
        [algebraic_base^(-i) for i in reverse(0:amount_layers-1)]
    elseif relaxation_method == :blurring
        [[max_blur * (1/i^2) for i in 1:amount_layers-1]; [0]]
    elseif relaxation_method == :none
        [1 for i in 1:amount_layers]
    else
        throw(error("Not implemented!"))
    end

    # sample from reference map
    X = sample_reference(reference_map, N_sample)
    # compute pdf
    Y = if broadcasted_tar_pdf
       π_tar(X, relax_param[1]) 
    else
        π_tar.(X, Ref(relax_param[1]))
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
        add_layer!(sra, layer_method, approx_method; 
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
    for j=reverse(_layers) # reverse order
        u = _ref_pushforward(sra, u)
        u = pushforward(sra.samplers[j], u)
    end
    return u
end

function pullback(
        sra::SelfReinforcedSampler{d, T}, 
        x::PSDdata{T};
        layers=nothing
    ) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.models)) : layers
    for j=_layers
        x = pullback(sra.samplers[j], x)
        x = _ref_pullback(sra, x)
    end
    return x
end