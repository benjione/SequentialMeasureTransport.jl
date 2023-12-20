

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
    println(io, "  reference map: $(sra.R_map)")
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

function _pullback_jacobian_function(
        sra::CondSampler{d, <:Any, T};
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
        sra::CondSampler{d, <:Any, T},
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

pushforward_pdf_function(sra::CondSampler; layers=nothing) = begin
    return pushforward_pdf_function(sra, x->reference_pdf(sra, x); layers=layers)
end
function pushforward_pdf_function(
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
    return marg_pushforward_pdf_function(sra, x->reference_pdf(sra, x); layers=layers)
end
function marg_pushforward_pdf_function(
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
    pushforward_pdf_function(sra, x->one(T))(x)
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

"""
Algorithms to create samplers
"""
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
                dC=0,
                kwargs...) where {d, T<:Number, S}

    fit_method! = if approx_method == :Chi2
        (m,x,y) -> Chi2_fit!(m, x, y; kwargs...)
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
        let π_tar=bridging_π
            (x, k) -> pushforward(reference_map, x->π_tar(x, k))(x)
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
    samplers = if dC==0
        [Sampler(model)]
    else
        [ConditionalSampler(model, dC)]
    end

    sra = if dC==0
        Sampler(samplers, reference_map)
    else
        CondSampler(samplers, reference_map)
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
        model::PSDModelOrthonormal{dsub, T, S},
        bridge::BridgingDensity{dsub, T},
        reference_map::ReferenceMap{d, T};
        subsample_data=false,
        subsample_size=2000,
        subspace_reference_map=nothing,
        to_subspace_reference_map=nothing,
        threading=true,
        dC=0,
        dCsub=0,
        kwargs...
) where {d, dsub, T<:Number, S}
    _d = length(X[1]) # data dimension
    @assert dsub ≤ d
    @assert _d == d

    if dsub < d
        @assert subspace_reference_map !== nothing
        @assert typeof(subspace_reference_map) <: ReferenceMap{dsub, T}
        if to_subspace_reference_map === nothing
            to_subspace_reference_map = reference_map
        end
        if dC > 0
            @assert dCsub > 0
            @assert dCsub ≤ dC
        end
    end

    L = domain_interval_left(model)
    R = domain_interval_right(model)
    @assert all(L .== 0.0)
    @assert all(R .== 1.0)
    if S<:OMF
        throw(error("Do not use OMF models for self reinforced sampler, use a Gaussian reference map instead!"))
    end

    sra = if dC == 0
        Sampler(Mapping{d, T}[], 
                                reference_map)
    else
        CondSampler(ConditionalMapping{d, dC, T}[], 
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
        layer = if dsub < d
            # select subset randomly
            B, P, P_tilde = if dC==0
                RandomSubsetProjection(T, d, dsub)
            else
                RandomConditionalSubsetProjection(T, 
                        d, dC, dsub, 
                        dCsub)
            end
            X_filter = [project_to_subset(P_tilde, 
                            to_subspace_reference_map, 
                            subspace_reference_map,
                            x) for x in X_evolved_pb]
            ML_fit!(model_ML, X_filter; kwargs...)
            sampler = if dC==0
                Sampler(model_ML)
            else
                ConditionalSampler(model_ML, dCsub)
            end
            ProjectionMapping{d, dC}(sampler, B, 
                                P, P_tilde, 
                                to_subspace_reference_map, 
                                subspace_reference_map)
        else
            ML_fit!(model_ML, X_evolved_pb; kwargs...)
            if dC==0
                Sampler(model_ML)
            else
                ConditionalSampler(model_ML, dC)
            end
        end
        add_layer!(sra, layer)
    end

    return sra
end

function Adaptive_Self_reinforced_ML_estimation(
    X_train::PSDDataVector{T},
    X_val::PSDDataVector{T},
    model::PSDModelOrthonormal{dsub, T, S},
    β::T,
    reference_map::ReferenceMap{d, T};
    ϵ=1e-3,
    subsample_data=false,
    subsample_size=2000,
    subspace_reference_map=nothing,
    to_subspace_reference_map=nothing,
    threading=true,
    dC=0,
    dCsub=0,
    kwargs...
) where {T<:Number, S, d, dsub}
    _d = length(X_train[1]) # data dimension
    @assert dsub ≤ d
    @assert _d == d

    bridge = BridgingDensities.DiffusionBrigdingDensity{d2, T}()

    if d2 < d
        @assert subspace_reference_map !== nothing
        @assert typeof(subspace_reference_map) <: ReferenceMap{d2, T}
        if to_subspace_reference_map === nothing
            to_subspace_reference_map = reference_map
        end
        if dC > 0
            @assert dCsub > 0
            @assert dCsub ≤ dC
        end
    end

    L = domain_interval_left(model)
    R = domain_interval_right(model)
    @assert all(L .== 0.0)
    @assert all(R .== 1.0)
    if S<:OMF
        throw(error("Do not use OMF models for self reinforced sampler, use a Gaussian reference map instead!"))
    end

    sra = if dC == 0
        Sampler(Mapping{d, T}[], 
                reference_map)
    else
        CondSampler(ConditionalMapping{d, dC, T}[], 
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
        layer = if dsub < d
            # select subset randomly
            B, P, P_tilde = if dC==0
                RandomSubsetProjection(T, d, dsub)
            else
                RandomConditionalSubsetProjection(T, 
                        d, dC, dsub, 
                        dCsub)
            end
            X_filter = [project_to_subset(P_tilde, 
                            to_subspace_reference_map, 
                            subspace_reference_map,
                            x) for x in X_evolved]
            ML_fit!(model_ML, X_filter; kwargs...)
            sampler = if dC==0
                Sampler(model_ML)
            else
                ConditionalSampler(model_ML, dCsub)
            end
            ProjectionMapping{d, dC}(sampler, B, 
                                P, P_tilde, 
                                to_subspace_reference_map, 
                                subspace_reference_map)
        else
            ML_fit!(model_ML, X_evolved; kwargs...)
            if dC==0
                Sampler(model_ML)
            else
                ConditionalSampler(model_ML, dC)
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
