module GraphicalModels
using ..PSDModels
using ..PSDModels: PSDdata
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
function (f::Factor{T})(x::Vector{T}) where {T<:Number}
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
function (m::GraphicalModel{T})(x::Vector{T}) where {T<:Number}
    return mapreduce(f->f(x), *, m.factors)
end


"""
Algorithm 
"""
function GraphSampler(π::GraphicalModel{T},
        model_factory::Function, # creates a model given dimension
        fit_method!::Function,
        R1_map::ReferenceMap{d, T},
        R2_map::ReferenceMap{d, T};
        N_sample=1000,
        threading=true,
    ) where {d, T<:Number}

    sampler = CondSampler{d, 0, T}(R1_map, R2_map)
    ## develop a parallelization technique
    for (k, fac) in enumerate(π.factors)
        dsub = length(fac.variables)
        model = model_factory(dsub)
        X = rand(T, d, N_sample)
        pb_coupling = pushforward(R2_map, pullback(sampler, x->fac(x)))
        Y = map(x->pb_coupling(x), eachcol(X))
        X_sub = X[fac.variables, :]
        fit_method!(model, eachcol(X_sub), Y)
        smp = Sampler(model)
        smp_sub = MarginalMapping{d, 0}(smp, fac.variables)
        PSDModels.add_layer!(sampler, smp_sub)
    end
    return sampler
end


end # module GraphicalModels