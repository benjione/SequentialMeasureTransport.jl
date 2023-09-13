
"""
Approximations T_1, T_2, ..., T_n of the target distribution π
so that
    T_1 ∘ T_2 ∘ ... ∘ T_n ∘ ρ = π
where ρ is the reference distribution.
All samplers are defined on the unit cube [0, 1]^d, so that
T_1 = R ∘ T_1' ∘ R^{-1} and
    T_1 ∘ T_2 ∘ ... ∘ T_n ∘ ρ = R ∘ T_1' ∘ ... ∘ T_n' ∘ R^{-1} ∘ ρ = π
"""
struct SelfReinforcedSampler{d, T, R} <: Sampler{d, T, R}
    samplers::Vector{<:Sampler{d, T}}   # defined on [0, 1]^d
    R_map::R    # reference map from reference distribution to uniform on [0, 1]^d
    function SelfReinforcedSampler(
        samplers::Vector{<:Sampler{d, T}},
        R_map::R
    ) where {d, T<:Number, R<:ReferenceMap}
        new{d, T, R}(samplers, R_map)
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
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    # apply map R (domain transformation)
    pdf_func = let sra=sra, pdf_func=pdf_tar
        u->begin
            pdf_func(_ref_pullback(sra, u)) * _ref_inv_Jacobian(sra, u)
        end
    end
    for sampler in sra.samplers[_layers]
        # apply map T_i
        pdf_func = let sra=sra, pdf_func=pdf_func, sampler=sampler
            u-> begin
                x = pushforward(sampler, u) # [0,1]->[a, b]
                return pdf_func(x) * (1/(Distributions.pdf(sampler, x)))
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
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    jacob_func = let jacob_func=jacob_func, sra=sra
        x->begin
            return jacob_func(_ref_pushforward(sra, x)) * _ref_Jacobian(sra, x)
        end
    end
    for j=_layers
        jacob_func = let jacob_func=jacob_func, sra=sra, j=j
            x->begin  
                x_for2 = pushforward(sra.samplers[j], x)
                return jacob_func(x_for2) * (1/(Distributions.pdf(sra.samplers[j], x_for2)))
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
    # from last to first
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    pdf_func = let sra=sra, pdf_func=pdf_ref
        u->begin
            pdf_func(_ref_pullback(sra, u)) * _ref_inv_Jacobian(sra, u)
        end
    end
    for sampler in reverse(sra.samplers[_layers])
        # apply map T_i
        pdf_func = let sra=sra, pdf_func=pdf_func, sampler=sampler
            x-> begin
                pdf_func(pullback(sampler, x)) * Distributions.pdf(sampler, x)
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
    push!(sra.samplers, Sampler(model))
    return nothing
end

function add_layer!(
        sra::SelfReinforcedSampler{d, T},
        sampler::Sampler{d, T},
    ) where {d, T<:Number}
    push!(sra.samplers, sampler)
    return nothing
end

function SelfReinforcedSampler(
            pdf_tar::Function,
            model::PSDModelOrthonormal{d, T, S},
            amount_layers::Int,
            approx_method::Symbol,
            reference_map::ReferenceMap{d, T};
            relaxation_method=:algebraic,
            ### for bridging densities
            N_sample=1000,
            max_blur=1.0,
            algebraic_base=2.0,
            N_relaxation=20, # number of MC for blurring
            langevin_time_grad=1.0,
            langevin_max_time=0.2,
            kwargs...) where {d, T<:Number, S}
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

    bridging_π = if relaxation_method == :algebraic
        AlgebraicBridgingDensity{d}(pdf_tar, relax_param)
    elseif relaxation_method == :blurring
        throw(error("Not implemented!"))
    else
        throw(error("Not implemented!"))
    end
    return SelfReinforcedSampler(bridging_π, model, amount_layers, approx_method, reference_map; 
            N_sample=N_sample, 
            kwargs...)
end
function SelfReinforcedSampler(
                bridging_π::BridgingDensity{d, T},
                model::PSDModelOrthonormal{d, T, S},
                amount_layers::Int,
                approx_method::Symbol,
                reference_map::ReferenceMap{d, T};
                ### for bridging densities
                N_sample=1000,
                ### others
                broadcasted_tar_pdf=false,
                threading=true,
                kwargs...) where {d, T<:Number, S}

    fit_method! = if approx_method == :Chi2
        @info "Chi2 will be soon deprecated!"
        (m,x,y) -> Chi2_fit!(m, x, y; kwargs...)
    elseif approx_method == :Chi2U
        (m,x,y) -> Chi2U_fit!(m, x, y; kwargs...)
    elseif approx_method == :Hellinger
        (m,x,y) -> Hellinger_fit!(m, x, y; kwargs...)
    elseif approx_method == :TV
        (m,x,y) -> TV_fit!(m, x, y; kwargs...)
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

    π_tar_samp = if broadcasted_tar_pdf
        let π_tar=bridging_π, reference_map=reference_map
            (x, k) -> π_tar(pullback.(Ref(reference_map), x), k) .* inverse_Jacobian.(Ref(reference_map), x)
        end
    else
        let π_tar=bridging_π, reference_map=reference_map
            (x, k) -> π_tar(pullback(reference_map, x), k) * inverse_Jacobian(reference_map, x)
        end
    end

    # sample from reference map
    X = eachcol(rand(T, d, N_sample))
    # compute pdf
    Y = if broadcasted_tar_pdf
        π_tar_samp(X, 1)
    else
        π_tar_samp.(X, Ref(1))
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

    sra = SelfReinforcedSampler(samplers, reference_map)

    for i in 2:amount_layers
        layer_method = let i=i, bridging_π=bridging_π
            x->bridging_π(x, i)
        end
        add_layer!(sra, layer_method, deepcopy(model), fit_method!; 
                N_sample=N_sample, 
                broadcasted_tar_pdf=broadcasted_tar_pdf,
                kwargs...)
    end
    
    return sra
end


function SelfReinforced_ML_estimation(
        X::PSDDataVector{T},
        model::PSDModelOrthonormal{d2, T, S},
        bridge::BridgingDensity{d2, T},
        reference_map::ReferenceMap{dr, T};
        subsample_data=false,
        subsample_size=2000,
        subspace_reference_map=nothing,
        to_subspace_reference_map=nothing,
        threading=true,
        kwargs...
) where {dr, d2, T<:Number, S}
    d = length(X[1]) # data dimension
    @assert d2 ≤ d
    @assert dr == d

    if d2 < d
        @assert subspace_reference_map !== nothing
        @assert typeof(subspace_reference_map) <: ReferenceMap{d2, T}
        if to_subspace_reference_map === nothing
            to_subspace_reference_map = reference_map
        end
    end

    L = domain_interval_left(model)
    R = domain_interval_right(model)
    @assert all(L .== 0.0)
    @assert all(R .== 1.0)
    if S<:OMF
        throw(error("Do not use OMF models for self reinforced sampler, use a Gaussian reference map instead!"))
    end

    sra = SelfReinforcedSampler(Sampler{d, T}[], 
                                reference_map)

    for t in bridge.t_vec
        @assert t ≥ 0.0
        model_ML = deepcopy(model)
    
        X_iter = if subsample_data
            StatsBase.sample(X, subsample_size, replace=false)
        else
            X
        end

        ## Create bridging densities according to method
        X_evolved = evolve_samples(bridge, X_iter, t)

        ## pullback and mapping to reference space
        X_evolved_pb = if length(sra.samplers)>0
            if threading
                map_threaded(x->_ref_pushforward(sra, pullback(sra, x)), X_evolved)
            else
                map(x->_ref_pushforward(sra, pullback(sra, x)), X_evolved)
            end
        else
            if threading
                map_threaded(x->_ref_pushforward(sra, x), X_evolved)
            else
                map(x->_ref_pushforward(sra, x), X_evolved)
            end
        end

        ## filter to the right dimensions
        if d2 < d
            B, P, P_tilde = RandomSubsetProjection(T, d, d2) # select subset randomly
            X_filter = [project_to_subset(P_tilde, 
                            to_subspace_reference_map, 
                            subspace_reference_map,
                            x) for x in X_evolved_pb]
            ML_fit!(model_ML, X_filter; kwargs...)
            layer = SubsetSampler{d}(Sampler(model_ML), B, 
                                P, P_tilde, 
                                to_subspace_reference_map, 
                                subspace_reference_map)
        else
            ML_fit!(model_ML, X_evolved_pb; kwargs...)
            layer = Sampler(model_ML)
        end

        add_layer!(sra, layer)
    end

    return sra
end

function pushforward(
        sra::SelfReinforcedSampler{d, T}, 
        u::PSDdata{T};
        layers=nothing
    ) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
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
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    x = _ref_pushforward(sra, x)
    for j=_layers
        x = pullback(sra.samplers[j], x)
    end
    x = _ref_pullback(sra, x)
    return x
end