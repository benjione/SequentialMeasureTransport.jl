
"""
Algorithms to create samplers
"""
function SelfReinforcedSampler(
            pdf_tar::Function,
            model::PSDModelOrthonormal{d, T, S},
            amount_layers::Int,
            approx_method::Symbol,
            reference_map::ReferenceMap{d, <:Any, T};
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
                reference_map::ReferenceMap{d, <:Any, T};
                ### for bridging densities
                N_sample=1000,
                custom_fit=nothing, # Function with arguments (model, X, Y) modifying model, can be created using minimize!
                ### others
                broadcasted_tar_pdf=false,
                threading=true,
                pdf_threading=true,     # turn off threading for pdf computation while keeping threading in general
                variable_ordering=nothing,
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
        @assert typeof(custom_fit) <: Function
        @info "Using custom fit method! Optional arguments for fitting method are ignored."
        (m,x,y) -> custom_fit(m, x, y)
    elseif approx_method == :adaptive
        (m,x,y,g) -> custom_fit(m, x, y, g; broadcasted_target=broadcasted_tar_pdf, kwargs...)
    else
        throw(error("Approx mehtod $(approx_method) not implemented!"))
    end

    L = domain_interval_left(model)
    R = domain_interval_right(model)
    @assert all(L .== 0.0)
    @assert all(R .== 1.0)
    @assert typeof(reference_map) <: ReferenceMap{d, dC, T}
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

    model_orig = model
    model = deepcopy(model)
    if approx_method == :adaptive
        X = Vector{T}[]
        Y = T[]
        g(x) = π_tar_samp(x, 1)
        fit_method!(model, X, Y, g)

    else
        # sample from reference map
        X = eachcol(rand(T, d, N_sample))
        # compute pdf
        Y = if broadcasted_tar_pdf
            π_tar_samp(X, 1)
        else
            _Y = zeros(T, N_sample)
            @_condusethreads threading&&pdf_threading for i in 1:N_sample
                _Y[i] = π_tar_samp(X[i], 1)
            end
            _Y
        end

        if any(isnan, Y)
            throw(error("NaN in target!"))
        end

        fit_method!(model, X, Y)
    end

    if any(isnan, model.B) || any(isinf, model.B)
        throw(error("NaN in model! Model did not converge!"))
    end

    normalize!(model)
    samplers = if dC==0
        if variable_ordering === nothing
            [Sampler(model)]
        else
            [Sampler(model, variable_ordering)]
        end
    else
        if variable_ordering === nothing
            [ConditionalSampler(model, dC)]
        else
            [ConditionalSampler(model, dC, variable_ordering)]
        end
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
        add_layer!(sra, layer_method, deepcopy(model_orig), fit_method!; 
                N_sample=N_sample, 
                broadcasted_tar_pdf=broadcasted_tar_pdf,
                threading=threading, variable_ordering=variable_ordering,
                approx_method=approx_method,
                kwargs...)
    end
    
    return sra
end

function add_layer!(
        sra::CondSampler{d, dC, T},
        pdf_tar::Function,
        model::PSDModelOrthonormal{d, T},
        fit_method!::Function;
        N_sample=1000,
        broadcasted_tar_pdf=false,
        threading=true,
        pdf_threading=true,     # turn off threading for pdf computation while keeping threading in general
        variable_ordering=nothing,
        approx_method=:Chi2,
        kwargs...
    ) where {d, T<:Number, dC}
    # sample from reference map
    pdf_tar_pullbacked = if broadcasted_tar_pdf
        _broadcasted_pullback_pdf_function(sra, pdf_tar)        
    else 
        pullback(sra, pdf_tar)
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
    if approx_method == :adaptive
        X = Vector{T}[]
        Y = T[]
        fit_method!(model, X, Y, x->pdf_tar_pullbacked_sample(x))
    else
        X = eachcol(rand(T, d, N_sample))
        Y = if broadcasted_tar_pdf
            pdf_tar_pullbacked_sample(X)
        else
            _Y = zeros(T, N_sample)
            @_condusethreads threading && pdf_threading for i in 1:N_sample
                _Y[i] = pdf_tar_pullbacked_sample(X[i])
            end
            _Y
        end

        if any(isnan, Y)
            throw(error("NaN in target!"))
        end

        fit_method!(model, collect(X), Y)
    end
    normalize!(model)
    if dC == 0
        if variable_ordering === nothing
            push!(sra.samplers, Sampler(model))
        else
            push!(sra.samplers, Sampler(model, variable_ordering))
        end
    else
        if variable_ordering === nothing
            push!(sra.samplers, ConditionalSampler(model, dC))
        else
            push!(sra.samplers, ConditionalSampler(model, dC, variable_ordering))
        end
    end
    return nothing
end

function add_layer!(
        sra::CondSampler{d, dC, T},
        pdf_tar::Function,
        model::PSDModelOrthonormal{d2, T},
        fit_method!::Function,
        subvariables::AbstractVector{Int};
        N_sample=1000,
        broadcasted_tar_pdf=false,
        threading=true,
        pdf_threading=true,     # turn off threading for pdf computation while keeping threading in general
        variable_ordering=nothing,
        dC2 = 0,
        kwargs...
    ) where {d, d2, T<:Number, dC}
    @assert d2 < d
    @assert length(subvariables) == d2
    @assert dC2 ≤ dC
    # sample from reference map
    X = rand(T, d, N_sample)
    pdf_tar_pullbacked = if broadcasted_tar_pdf
        _broadcasted_pullback_pdf_function(sra, pdf_tar)        
    else 
        pullback(sra, (x)->pdf_tar(x[subvariables]))
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
        pdf_tar_pullbacked_sample(eachcol(X))
    else
        _Y = zeros(T, N_sample)
        _X = eachcol(X)
        @_condusethreads threading && pdf_threading for i in 1:N_sample
            _Y[i] = pdf_tar_pullbacked_sample(_X[i])
        end
        _Y
    end

    if any(isnan, Y)
        throw(error("NaN in target!"))
    end

    fit_method!(model, collect(eachcol(X[subvariables, :])), Y)
    normalize!(model)
    sampler = if dC2 == 0
        if variable_ordering === nothing
            Sampler(model)
        else
            Sampler(model, variable_ordering)
        end
    else
        if variable_ordering === nothing
            ConditionalSampler(model, dC2)
        else
            ConditionalSampler(model, dC2, variable_ordering)
        end
    end
    push!(sra.samplers, MarginalMapping{d, dC}(sampler, subvariables))
    return nothing
end


function SelfReinforced_ML_estimation(
        X::PSDDataVector{T},
        model::PSDModelOrthonormal{dsub, T, S},
        bridge::BridgingDensity{d, T},
        reference_map::ReferenceMap{d, <:Any, T};
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
    @assert typeof(reference_map) <: ReferenceMap{d, dC, T}

    if dsub < d
        @assert subspace_reference_map !== nothing
        @assert typeof(subspace_reference_map) <: ReferenceMap{dsub, dCsub, T}
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
                map_threaded(typeof(X_evolved[1]), x->_ref_pushforward(sra, pullback(sra, x)), X_evolved)
            else
                map(x->_ref_pushforward(sra, pullback(sra, x)), X_evolved)
            end
        else
            if threading
                map_threaded(typeof(X_evolved[1]), x->_ref_pushforward(sra, x), X_evolved)
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


"""
Adaptive self-reinforced maximum likelihood estimation

This function creates a Sampler by adding layers of bridging densities
until the residual of the sampler is larger than the smallest residual times (1+ϵ_smallest) or
larger than the previous residual times (1 + ϵ_last).

# Arguments
- `X_train::PSDDataVector{T}`: Training data
- `X_val::PSDDataVector{T}`: Validation data
- `model::PSDModelOrthonormal{dsub, T, S}`: Model to be fitted
- `β::T`: Bridging density parameter
- `reference_map::ReferenceMap{d, <:Any, T}`: Reference map
- `ϵ_last=1e-4`: Residual threshold for previous residual
- `ϵ_smallest=1e-3`: Residual threshold for smallest residual
- `return_smallest_residual=false`: If true, after conditions is reached the last layers are removed and the sampler with the smallest residual is returned
- `residual`: Residual function, default is negative log likelihood
- `L_max=100`: Maximum number of layers
- `subsample_data=false`: Subsample data in case of large data
- `subsample_size=2000`: Size of subsample, only used if `subsample_data=true`
- `threading=true`: Enable/disable threading
- `dC=0`: Conditional dimension to estimate π(y | x) where y are the last dC dimensions
- `dCsub=0`: Conditional dimension of the subspace, at least 1 if `dsub < d` and `dC > 0`
- `trace_df=nothing`: DataFrame to store trace information. If not `nothing`, the following columns are added at every iteration:
    - `:residual`: Residual of the sampler
    - `:layers`: Number of layers
    - `:time`: Time to add layer
"""
function Adaptive_Self_reinforced_ML_estimation(
    X_train::PSDDataVector{T},
    X_val::PSDDataVector{T},
    model::PSDModelOrthonormal{dsub, T, S},
    β::T,
    reference_map::ReferenceMap{d, <:Any, T};
    ϵ_last=1e-4,
    ϵ_smallest=1e-3,
    return_smallest_residual=false,
    residual=nothing,
    L_max=100,
    subsample_data=false,
    subsample_size=2000,
    subspace_reference_map=nothing,
    to_subspace_reference_map=nothing,
    threading=true,
    dC=0,
    dCsub=0,
    trace_df=nothing,
    σ=nothing,
    μ=nothing,
    kwargs...
) where {T<:Number, S, d, dsub}
    _d = length(X_train[1]) # data dimension
    @assert dsub ≤ d
    @assert _d == d
    @assert typeof(reference_map) <: ReferenceMap{d, dC, T}

    bridge = if σ === nothing
        BridgingDensities.DiffusionBrigdingDensity{d, T}()
    else
        @assert μ !== nothing
        BridgingDensities.DiffusionBrigdingDensity{d, T}(σ, μ)
    end

    if dsub < d
        @assert subspace_reference_map !== nothing
        @assert typeof(subspace_reference_map) <: ReferenceMap{dsub, dCsub, T}
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

    # residual as negative log likelihood
    _residual(mapping, X_val) = if residual === nothing 
        (1/length(X_val)) * mapreduce(x->-Distributions.logpdf(mapping, x), +, X_val)
    else
        residual(mapping, X_val)
    end
    last_residual = Inf64
    smallest_residual = Inf64

    while length(sra.samplers) < L_max
        start_time = time()
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

        resid = _residual(sra, X_val)
        if (resid - smallest_residual) > ϵ_smallest || (resid - last_residual) > ϵ_last
            pop!(sra.samplers)
            break
        end
        last_residual = resid
        if resid < smallest_residual
            smallest_residual = resid
        end

        end_time = time()

        ## store trace information
        if trace_df !== nothing
            push!(trace_df, (residual=resid, layers=length(sra.samplers), time=end_time-start_time))
        end

        X_evolved = nothing
        X_iter = nothing
        GC.gc()
    end

    if return_smallest_residual
        ## calc residual after each layer
        _resids = zeros(T, length(sra.samplers))
        for i in 1:length(sra.samplers)
            _sra = CondSampler(sra.samplers[1:i], sra.R1_map, sra.R2_map)
            _resids[i] = _residual(_sra, X_val)
        end
        ## find layer with smallest residual
        min_idx = argmin(_resids)
        return CondSampler(sra.samplers[1:min_idx], sra.R1_map, sra.R2_map)
    end

    return sra
end
