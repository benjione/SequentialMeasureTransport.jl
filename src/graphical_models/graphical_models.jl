


struct Factor{T<:Number}
    variables::Vector{Int}
    coupling::Function
end
function (f::Factor{T})(x::Vector{T}) where {T<:Number}
    if length(x) == length(f.variables)
        return f.coupling(x)
    end
    return f.coupling(x[f.variables])
end


struct GraphicalModel{T}
    factors::Vector{Factor{T}}
    independency_matrix::Matrix{Bool}          # independency matrix, true for dependent
    function GraphicalModel{T}(factors::Vector{Factor{T}}) where {T<:Number}
        independency_matrix = [length(intersect(Set(f1.variables), Set(f2.variables))) == 0 for f1 in factors, f2 in factors]
        new{T}(factors, independency_matrix)
    end
end
function (m::GraphicalModel{T})(x::Vector{T}) where {T<:Number}
    return mapreduce(f->f(x), *, m.factors)
end

mutable struct GraphSampler{d, T<:Number, R, dC} <: ConditionalSampler{d, T, R, dC}
    samplers::Vector{SubsetSampler{d, <:Any, T, <:Any, <:Any, dC, <:Any, <:Any}}
    R_map::R
    function GraphSampler{d, T<:Number, R, dC}(R_map::R) where {d, T<:Number, R, dC}
        new{d, T, R, dC}(SubsetSampler{d, <:Any, T, <:Any, <:Any, dC, <:Any, <:Any}[], R_map)
    end
end

function pullback(sampler::GraphSampler{d, T, R, dC}, u::PSDdata{T}) where {d, T<:Number, R, dC}
    x = foldl((s, u)->pullback(s, u), sampler.samplers, init=pullback(sampler.R_map, u))
    return pushforward(sampler.R_map, x)
end

function pushforward(sampler::GraphSampler{d, T, R, dC}, x::PSDdata{T}) where {d, T<:Number, R, dC}
    u = foldr((s, x)->pushforward(s, x), sampler.samplers, init=pullback(sampler.R_map, x))
    return pushforward(sampler.R_map, u)
end

# function Jacobian(sampler::GraphSampler{d, T, R, dC}, x::PSDdata{T}) where {d, T<:Number, R, dC}
#     return mapreduce(s->Jacobian(s, x), *, sampler.samplers)
# end

# function inverse_Jacobian(sampler::GraphSampler{d, T, R, dC}, u::PSDdata{T}) where {d, T<:Number, R, dC}
#     return mapreduce(s->inverse_Jacobian(s, u), *, sampler.samplers)
# end

function GraphSampler(π::GraphicalModel{T},
        model::PSDModelOrthonormal{d, T, S},
        fit_method!::Function,
        R_map::ReferenceMap{d, T};
        N_sample=1000,
        threading=true,
    ) where {d, T<:Number, S}

    graph_sampler = GraphSampler{d, T, ReferenceMap{d, T}, d}(R_map)
    for (fac, k) in enumerate(π.factors)
        X = eachcol(rand(T, length(fac.variables), N_sample))
        Y = map(x->fac(x), X)
        model_tmp = deepcopy(model)
        fit_method!(model_tmp, X, Y)
        smp = Sampler(model_tmp)
        push!(list_mappings, smp)
    end

end

function SelfReinforcedSampler(
                π::GraphicalModel{T},
                model::PSDModelOrthonormal{d, T, S},
                fit_method!::Function,
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

    L = domain_interval_left(model)
    R = domain_interval_right(model)
    @assert all(L .== 0.0)
    @assert all(R .== 1.0)
    if S<:OMF
        throw(error("Do not use OMF models for self reinforced sampler, use a Gaussian reference map instead!"))
    end

    for k = 1:length(π.factors)
        @assert length(π.factors[k].variables) == 1
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