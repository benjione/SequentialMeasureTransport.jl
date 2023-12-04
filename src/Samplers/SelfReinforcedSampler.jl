

"""
Approximations T_1, T_2, ..., T_n of the target distribution π
so that
    T_1 ∘ T_2 ∘ ... ∘ T_n ∘ ρ = π
where ρ is the reference distribution.
All samplers are defined on the unit cube [0, 1]^d, so that
T_1 = R ∘ T_1' ∘ R^{-1} and
    T_1 ∘ T_2 ∘ ... ∘ T_n ∘ ρ = R ∘ T_1' ∘ ... ∘ T_n' ∘ R^{-1} ∘ ρ = π
"""
struct CondSelfReinforcedSampler{d, T, R, dC} <: ConditionalSampler{d, T, R, dC}
    samplers::Vector{<:ConditionalSampler{d, T, Nothing, dC}}   # defined on [0, 1]^d
    R_map::R    # reference map from reference distribution to uniform on [0, 1]^d
    function CondSelfReinforcedSampler(
        samplers::Vector{<:ConditionalSampler{d, T, Nothing, dC}},
        R_map::R,
    ) where {d, T<:Number, R<:ReferenceMap, dC}
        new{d, T, R, dC}(samplers, R_map)
    end
end


# Not a conditional sampler
const SelfReinforcedSampler{d, T, R} = CondSelfReinforcedSampler{d, T, R, 0}

function SelfReinforcedSampler(
            samplers::Vector{<:ConditionalSampler{d, T, Nothing, 0}},
            R_map::R;
        ) where {d, T<:Number, R<:ReferenceMap}
    return CondSelfReinforcedSampler(samplers, R_map)
end

## Pretty printing
function Base.show(io::IO, sra::CondSelfReinforcedSampler{d, T}) where {d, T<:Number}
    println(io, "SelfReinforcedSampler{d=$d, T=$T}")
    println(io, "  samplers:")
    for (i, sampler) in enumerate(sra.samplers)
        if i>3
            println(io, "...")
            break
        end
        println(io, "    $i: $sampler")
    end
    println(io, "  reference map: $(sra.R_map)")
end

## Overwrite pdf function from Distributions
function Distributions.pdf(
        sar::CondSelfReinforcedSampler{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    pdf_func = pushforward_pdf_function(sar)
    return pdf_func(x)
end

function pullback_pdf_function(
        sra::CondSelfReinforcedSampler{d, T},
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
        pdf_func = pullback(sampler, pdf_func)
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
        sra::CondSelfReinforcedSampler{d, T};
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
        sra::CondSelfReinforcedSampler{d, T},
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

pushforward_pdf_function(sra::CondSelfReinforcedSampler; layers=nothing) = begin
    return pushforward_pdf_function(sra, x->reference_pdf(sra, x); layers=layers)
end
function pushforward_pdf_function(
        sra::CondSelfReinforcedSampler{d, T},
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
        pdf_func = pushforward(sampler, pdf_func)
    end
    pdf_func = let sra=sra, pdf_func=pdf_func
        x->begin
            pdf_func(_ref_pushforward(sra, x)) * _ref_Jacobian(sra, x) # [a, b]^d -> [0, 1]^d
        end
    end
    return pdf_func
end


marg_pushforward_pdf_function(sra::CondSelfReinforcedSampler; layers=nothing) = begin
    return marg_pushforward_pdf_function(sra, x->reference_pdf(sra, x); layers=layers)
end
function marg_pushforward_pdf_function(
        sra::CondSelfReinforcedSampler{d, T},
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
                pdf_func(marg_pullback(sampler, x)) * marg_pdf(sampler, x)
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
        sra::CondSelfReinforcedSampler{d, T, <:Any, dC},
        pdf_tar::Function,
        model::PSDModelOrthonormal{d, T},
        fit_method!::Function;
        N_sample=1000,
        broadcasted_tar_pdf=false,
        threading=true,
        kwargs...
    ) where {d, T<:Number, dC}
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
        _Y = zeros(T, N_sample)
        @_condusethreads threading for i in 1:N_sample
            _Y[i] = pdf_tar_pullbacked_sample(X[i])
        end
        _Y
    end

    if any(isnan, Y)
        throw(error("NaN in target!"))
    end

    fit_method!(model, collect(X), Y)
    normalize!(model)
    if dC == 0
        push!(sra.samplers, Sampler(model))
    else
        push!(sra.samplers, ConditionalSampler(model, dC))
    end
    return nothing
end

function add_layer!(
        sra::CondSelfReinforcedSampler{d, T, <:Any, dC},
        sampler::ConditionalSampler{d, T, Nothing, dC},
    ) where {d, T<:Number, dC}
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
                custom_fit=nothing, # Function with arguments (model, X, Y) modifying model, can be created using minimize!
                ### others
                broadcasted_tar_pdf=false,
                threading=true,
                amount_cond_variable=0,
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
    elseif approx_method == :KL
        (m,x,y) -> KL_fit!(m, x, y; kwargs...)
    elseif approx_method == :custom
        @info "Using custom fit method!"
        @assert typeof(custom_fit) <: Function
        (m,x,y) -> custom_fit(m, x, y; kwargs...)
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
        _Y = zeros(T, N_sample)
        @_condusethreads threading for i in 1:N_sample
            _Y[i] = π_tar_samp(X[i], 1)
        end
        _Y
    end

    if any(isnan, Y)
        throw(error("NaN in target!"))
    end

    fit_method!(model, X, Y)

    if any(isnan, model.B)
        throw(error("NaN in model! Model did not converge!"))
    end

    normalize!(model)
    samplers = if amount_cond_variable==0
        [Sampler(model)]
    else
        [ConditionalSampler(model, amount_cond_variable)]
    end

    sra = if amount_cond_variable==0
        SelfReinforcedSampler(samplers, reference_map)
    else
        CondSelfReinforcedSampler(samplers, reference_map)
    end

    for i in 2:amount_layers
        layer_method = let i=i, bridging_π=bridging_π
            x->bridging_π(x, i)
        end
        add_layer!(sra, layer_method, deepcopy(model), fit_method!; 
                N_sample=N_sample, 
                broadcasted_tar_pdf=broadcasted_tar_pdf,
                threading=threading,
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
        amount_cond_variable=0,
        amount_reduced_cond_variables=0,
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
        if amount_cond_variable > 0
            @assert amount_reduced_cond_variables > 0
            @assert amount_reduced_cond_variables ≤ amount_cond_variable
        end
    end

    L = domain_interval_left(model)
    R = domain_interval_right(model)
    @assert all(L .== 0.0)
    @assert all(R .== 1.0)
    if S<:OMF
        throw(error("Do not use OMF models for self reinforced sampler, use a Gaussian reference map instead!"))
    end

    sra = if amount_cond_variable == 0
        SelfReinforcedSampler(Sampler{d, T, Nothing}[], 
                                reference_map)
    else
        CondSelfReinforcedSampler(ConditionalSampler{d, T, Nothing, amount_cond_variable}[], 
                                reference_map)
    end

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
        layer = if d2 < d
            # select subset randomly
            B, P, P_tilde = if amount_cond_variable==0
                RandomSubsetProjection(T, d, d2)
            else
                RandomConditionalSubsetProjection(T, 
                        d, amount_cond_variable, d2, 
                        amount_reduced_cond_variables)
            end
            X_filter = [project_to_subset(P_tilde, 
                            to_subspace_reference_map, 
                            subspace_reference_map,
                            x) for x in X_evolved_pb]
            ML_fit!(model_ML, X_filter; kwargs...)
            sampler = if amount_cond_variable==0
                Sampler(model_ML)
            else
                ConditionalSampler(model_ML, amount_reduced_cond_variables)
            end
            SubsetSampler{d, amount_cond_variable}(sampler, B, 
                                P, P_tilde, 
                                to_subspace_reference_map, 
                                subspace_reference_map)
        else
            ML_fit!(model_ML, X_evolved_pb; kwargs...)
            if amount_cond_variable==0
                Sampler(model_ML)
            else
                ConditionalSampler(model_ML, amount_cond_variable)
            end
        end
        add_layer!(sra, layer)
    end

    return sra
end

function Adaptive_Self_reinforced_ML_estimation(
    X_train::PSDDataVector{T},
    X_val::PSDDataVector{T},
    model::PSDModelOrthonormal{d2, T, S},
    β::T,
    reference_map::ReferenceMap{dr, T};
    ϵ=1e-3,
    subsample_data=false,
    subsample_size=2000,
    subspace_reference_map=nothing,
    to_subspace_reference_map=nothing,
    threading=true,
    amount_cond_variable=0,
    amount_reduced_cond_variables=0,
    kwargs...
) where {T<:Number, S, dr, d2}
    d = length(X_train[1]) # data dimension
    @assert d2 ≤ d
    @assert dr == d

    bridge = BridgingDensities.DiffusionBrigdingDensity{d2, T}()

    if d2 < d
        @assert subspace_reference_map !== nothing
        @assert typeof(subspace_reference_map) <: ReferenceMap{d2, T}
        if to_subspace_reference_map === nothing
            to_subspace_reference_map = reference_map
        end
        if amount_cond_variable > 0
            @assert amount_reduced_cond_variables > 0
            @assert amount_reduced_cond_variables ≤ amount_cond_variable
        end
    end

    L = domain_interval_left(model)
    R = domain_interval_right(model)
    @assert all(L .== 0.0)
    @assert all(R .== 1.0)
    if S<:OMF
        throw(error("Do not use OMF models for self reinforced sampler, use a Gaussian reference map instead!"))
    end

    sra = if amount_cond_variable == 0
        SelfReinforcedSampler(Sampler{d, T, Nothing}[], 
                                reference_map)
    else
        CondSelfReinforcedSampler(ConditionalSampler{d, T, Nothing, amount_cond_variable}[], 
                                reference_map)
    end

    Residual(mapping) = mapreduce(x->-log(Distributions.pdf(mapping, x)), +, X_val)
    last_residual = Inf64

    while true
        t_ℓ = BridgingDensities.add_timestep!(bridge, β)
        model_ML = deepcopy(model)
    
        X_iter = if subsample_data
            StatsBase.sample(X_train, subsample_size, replace=false)
        else
            X_train
        end

        ## Create bridging densities according to method
        X_evolved = evolve_samples(bridge, X_iter, t_ℓ)

        ## pullback and mapping to reference space
        X_evolved = if length(sra.samplers)>0
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
        layer = if d2 < d
            # select subset randomly
            B, P, P_tilde = if amount_cond_variable==0
                RandomSubsetProjection(T, d, d2)
            else
                RandomConditionalSubsetProjection(T, 
                        d, amount_cond_variable, d2, 
                        amount_reduced_cond_variables)
            end
            X_filter = [project_to_subset(P_tilde, 
                            to_subspace_reference_map, 
                            subspace_reference_map,
                            x) for x in X_evolved]
            ML_fit!(model_ML, X_filter; kwargs...)
            sampler = if amount_cond_variable==0
                Sampler(model_ML)
            else
                ConditionalSampler(model_ML, amount_reduced_cond_variables)
            end
            SubsetSampler{d, amount_cond_variable}(sampler, B, 
                                P, P_tilde, 
                                to_subspace_reference_map, 
                                subspace_reference_map)
        else
            ML_fit!(model_ML, X_evolved; kwargs...)
            if amount_cond_variable==0
                Sampler(model_ML)
            else
                ConditionalSampler(model_ML, amount_cond_variable)
            end
        end
        add_layer!(sra, layer)

        residual = Residual(sra)
        if residual > (1.0 + ϵ)* last_residual
            pop!(sra.samplers)
            break
        end
        last_residual = residual
        X_evolved = nothing
        X_iter = nothing
        GC.gc()
    end

    return sra
end

## Interface of Sampler

function pushforward(
        sra::CondSelfReinforcedSampler{d, T}, 
        u::PSDdata{T};
        layers=nothing
    ) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    u = _ref_pushforward(sra, u)
    for j=reverse(_layers) # reverse order
        u = pushforward(sra.samplers[j], u)
    end
    u = _ref_pullback(sra, u)
    return u::Vector{T}
end

function pullback(
        sra::CondSelfReinforcedSampler{d, T}, 
        x::PSDdata{T};
        layers=nothing
    ) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    x = _ref_pushforward(sra, x)
    for j=_layers
        x = pullback(sra.samplers[j], x)
    end
    x = _ref_pullback(sra, x)
    return x::Vector{T}
end

"""
TODO: can be done prettier
"""
function inverse_Jacobian(sra::CondSelfReinforcedSampler{d, T},
            x::PSDdata{T}) where {d, T<:Number}
    pushforward_pdf_function(sra, x->one(T))(x)
end
@inline Jacobian(sra::CondSelfReinforcedSampler{d, T},
            x::PSDdata{T}) where {d, T<:Number} = 1/inverse_Jacobian(sra, pushforward(sra, x))

## Interface of ConditionalSampler

function marg_pdf(sra::CondSelfReinforcedSampler{d, T, R, dC}, x::PSDdata{T}) where {d, T<:Number, R, dC}
    @assert length(x) == _d_marg(sra)
    pdf_func = marg_pushforward_pdf_function(sra)
    return pdf_func(x)
end

function marg_pushforward(sra::CondSelfReinforcedSampler{d, T}, u::PSDdata{T};
                        layers=nothing) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    u = _ref_pushforward(sra, u)
    for j=reverse(_layers) # reverse order
        u = marg_pushforward(sra.samplers[j], u)
    end
    u = _ref_pullback(sra, u)
    return u
end

function marg_pullback(sra::CondSelfReinforcedSampler{d, T}, x::PSDdata{T};
                        layers=nothing) where {d, T<:Number}
    _layers = layers === nothing ? (1:length(sra.samplers)) : layers
    x = _ref_pushforward(sra, x)
    for j=_layers
        x = marg_pullback(sra.samplers[j], x)
    end
    x = _ref_pullback(sra, x)
    return x
end