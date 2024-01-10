module GraphicalModels
using ..PSDModels
using ..PSDModels: PSDdata, PSDDataVector
using ..PSDModels: AbstractSampler
using ..PSDModels: PSDModelOrthonormal
using ..PSDModels: ReferenceMap, CondSampler, MarginalMapping
using ..PSDModels: pushforward, pullback

struct Factor{T<:Number}
    variables::Vector{Int}
    coupling::Function
    function Factor{T}(variables::Vector{Int}, coupling::Function) where {T<:Number}
        new{T}(variables, coupling)
    end
end
function (f::Factor{T})(x::PSDdata{T}) where {T<:Number}
    if length(x) == length(f.variables)
        return f.coupling(x)
    end
    return f.coupling(x[f.variables])
end


struct GraphicalModel{T}
    factors::Vector{Factor{T}}
    independency_matrix::Matrix{Bool}          # independency matrix, true for dependent
    function GraphicalModel(factors::Vector{Factor{T}}) where {T<:Number}
        independency_matrix = [length(intersect(Set(f1.variables), Set(f2.variables))) == 0 for f1 in factors, f2 in factors]
        new{T}(factors, independency_matrix)
    end
end
function (m::GraphicalModel{T})(x::PSDdata{T}) where {T<:Number}
    return mapreduce(f->f(x), *, m.factors)
end

struct TimeSeriesModel{T<:Number}
    π_X::Function                   # initial distribution, π_X(x_0)
    π_Θ::Function                   # prior distribution, π_Θ(θ)
    ForwardDistribution::Function   # forward model, f(x_i | x_{i-1}, θ)
    Likelihood::Function            # likelihood function, L(y_i | x_i)
    dx::Int                         # dimension of state space
    dy::Int                         # dimension of observation space
    dΘ::Int                         # dimension of parameter space
    N::Int                          # number of time steps
end

function TimeSeriesSampler(
        time_serie::TimeSeriesModel{T},
        Y::PSDDataVector{T},     # Observations
        model_factory::Function, # creates a model given dimension
        fit_method!::Function,
        R1_map::ReferenceMap{d, T},
        R2_map::ReferenceMap{d, T};
        N_sample=1000,
        bridging=nothing,
        threading::Bool=true,
        trace=false
    ) where {d, T<:Number}
    @assert (time_serie.dx * time_serie.N + time_serie.dΘ) == d
    dC = time_serie.dx * time_serie.N
    # dC = 0
    dmodel = 2*time_serie.dx + time_serie.dΘ
    dCmodel = time_serie.dx
    # dCmodel = 0
    sampler = CondSampler{d, dC, T}(R1_map, R2_map)
    ## develop a parallelization technique
    for k=1:time_serie.N-1
        # list of variables for the k-th time step
        var_param = 1:time_serie.dΘ
        var_state = (k*time_serie.dx+time_serie.dΘ+1):((k+1)*time_serie.dx+time_serie.dΘ)
        var_state_prev = ((k-1)*time_serie.dx+time_serie.dΘ+1):((k)*time_serie.dx+time_serie.dΘ)
        variables = [var_param; var_state; var_state_prev]

        if trace
            println("iteration: ", k)
            println("variables: ", variables)
        end

        # Coupling given by ϕ(x_{k}, x_{k-1}, θ) = f(x_{k}, x_{k-1}, θ) * L(y_k | x_k)
        coupling = if k>1
            (x) -> time_serie.ForwardDistribution(x[var_state], x[var_state_prev], x[var_param]) * 
                time_serie.Likelihood(Y[k], x[var_state], x[var_param])
        else
            (x) -> time_serie.π_X(x[var_state_prev]) *
                time_serie.π_Θ(x[var_param]) *
                time_serie.ForwardDistribution(x[var_state], x[var_state_prev], x[var_param]) *
                time_serie.Likelihood(Y[k], x[var_state], x[var_param])
        end

        for j=1:3
            alg_bridge_param = 2.0^(-3+j)
            model = model_factory(dmodel)
            # evaluate X = T(u), u ∼ R1
            X = rand(T, d, N_sample)

            # cut out the only variables we are interested in
            X_sub = X[variables, :]

            # use ϕ(T(x_{k}, x_{k-1}, θ))
            bridge_coupling = let coupling=coupling
                (x) -> coupling(pushforward(sampler, x))^alg_bridge_param
            end
            pb_coupling = pullback(R2_map, bridge_coupling)
            
            # evaluate Y = ϕ(X)
            Y = if threading
                PSDModels.map_threaded(T, x->pb_coupling(x), eachcol(X)) 
            else
                map(x->pb_coupling(x), eachcol(X))
            end

            fit_method!(model, eachcol(X_sub), Y; trace=trace)
            smp = ConditionalSampler(model, dCmodel)
            smp_sub = MarginalMapping{d, dC}(smp, variables)
            PSDModels.add_layer!(sampler, smp_sub)
        end
    end
    return sampler
end

"""
Algorithm 
"""
function GraphSampler(π::GraphicalModel{T},
        model_factory::Function, # creates a model given dimension
        fit_method!::Function,
        R1_map::ReferenceMap{d, T},
        R2_map::ReferenceMap{d, T};
        N_sample=2000,
        threading::Bool=true,
        trace=false,
    ) where {d, T<:Number}

    sampler = CondSampler{d, 0, T}(R1_map, R2_map)
    ## develop a parallelization technique
    for (k, fac) in enumerate(π.factors)
        dsub = length(fac.variables)
        model = model_factory(dsub)
        # X = rand(T, d, N_sample)

        # evaluate X = T(u), u ∼ R1
        X = rand(T, d, N_sample)

        # pb_coupling = pushforward(R2_map, pullback(sampler, x->fac(x)))
        # Y = map(x->pb_coupling(x), eachcol(X))
        # X_sub = X[fac.variables, :]
        pb_coupling = pullback(R2_map, x->fac(pushforward(sampler, x)[fac.variables]))
        Y = if threading
            PSDModels.map_threaded(T, x->pb_coupling(x), eachcol(X)) 
        else
            map(x->pb_coupling(x), eachcol(X))
        end

        X_sub = X[fac.variables, :]
        fit_method!(model, eachcol(X_sub), Y; trace=trace)
        smp = Sampler(model)
        smp_sub = MarginalMapping{d, 0}(smp, fac.variables)
        PSDModels.add_layer!(sampler, smp_sub)
    end
    return sampler
end


end # module GraphicalModels