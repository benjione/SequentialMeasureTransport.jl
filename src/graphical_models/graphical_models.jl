module GraphicalModels
using ..PSDModels
using ..PSDModels: PSDdata
using ..PSDModels: AbstractSampler, SubsetMapping
using ..PSDModels: PSDModelOrthonormal
using ..PSDModels: ReferenceMap
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

# mutable struct GraphMapping{d, dC, T<:Number} <: ConditionalMapping{d, dC, T}
#     mappings::Vector{<:SubsetMapping{d, dC, T}}
#     function GraphSampler{d, dC, T}() where {d, dC, T<:Number}
#         new{d, dC, T}(SubsetMapping{d, dC, T}[])
#     end
# end

# function PSDModels.pullback(sampler::GraphMapping{d, dC, T}, 
#                 u::PSDdata{T}
#         ) where {d, T<:Number, dC}
#     return foldl((s, u)->pullback(s, u), sampler.samplers, init=u)
# end

# function PSDModels.pushforward(sampler::GraphMapping{d, dC, T}, x::PSDdata{T}) where {d, dC, T}
#     return foldr((s, x)->pushforward(s, x), sampler.samplers, init=x)
# end

# function PSDModels.pushforward(sampler::GraphMapping{d, dC, T}, 
#             func::Function) where {d, T<:Number, R, dC}
#     func_res = foldr((s, f)->pushforward(s, f), sampler.samplers, init=pushforward(sampler.R_map, func))
#     return pullback(sampler.R_map, func_res)
# end

# function PSDModels.inverse_Jacobian(sampler::GraphMapping{d, dC, T}, x::PSDdata{T}) where {d, dC, T}
#     return pushforward(sampler.R_map, x->one(T))(x)
# end

# function PSDModels.Jacobian(sampler::GraphMapping{d, dC, T}, u::PSDdata{T}) where {d, dC, T}
#     return 1/inverse_Jacobian(sampler, pushforward(sampler, u))
# end


"""
Algorithm 
"""
function GraphSampler(π::GraphicalModel{T},
        model::PSDModelOrthonormal{d2, T},
        fit_method!::Function,
        R_map::ReferenceMap{d, T};
        N_sample=1000,
        threading=true,
    ) where {d, d2, T<:Number}

    graph_sampler = GraphSampler{d, T, ReferenceMap{d, T}, d}(R_map)
    for (k, fac) in enumerate(π.factors)
        X = eachcol(rand(T, length(fac.variables), N_sample))
        Y = map(x->pushforward(R_map, pullback(graph_sampler, x->fac(x)))(x), X)
        model_tmp = deepcopy(model)
        fit_method!(model_tmp, X, Y)
        smp = Sampler(model_tmp)
        smp_sub = SubsetSampler{d}(smp, fac.variables, R_map, R_map)
        push!(graph_sampler.samplers, smp_sub)
    end
    return graph_sampler
end


end # module GraphicalModels