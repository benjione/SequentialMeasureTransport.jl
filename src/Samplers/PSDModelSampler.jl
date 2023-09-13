using Roots: find_zero

struct PSDModelSampler{d, T<:Number, S, R} <: Sampler{d, T, R}
    model::PSDModelOrthonormal{d, T, S}                 # model to sample from
    margins::Vector{<:PSDModelOrthonormal{<:Any, T, S}} # start with x_{≤1}, then x_{≤2}, ...
    integrals::Vector{<:OrthonormalTraceModel{T, S}}    # integrals of marginals
    variable_ordering::AbstractVector{Int}              # variable ordering for the model
    R_map::R                                            # reference map from reference distribution to uniform
    function PSDModelSampler(model::PSDModelOrthonormal{d, T, S}, 
                             variable_ordering::AbstractVector{Int}) where {d, T<:Number, S}
        model = normalize(model) # create normalized copy
        margins = PSDModelOrthonormal{<:Any, T, S}[marginalize(model, variable_ordering[k:end]) for k in 2:d]
        margins = [margins; model] # add the full model as last
        integrals = map((x,k)->integral(x, k), margins, variable_ordering)
        new{d, T, S, Nothing}(model, margins, integrals, variable_ordering, nothing)
    end
end

Sampler(model::PSDModelOrthonormal{d}) where {d} = PSDModelSampler(model, collect(1:d))

function Distributions.pdf(
        sar::PSDModelSampler{d, T},
        x::PSDdata{T}
    ) where {d, T<:Number}
    return sar.model(x)
end

function pushforward(sampler::PSDModelSampler{d, T, S}, u::PSDdata{T}) where {d, T<:Number, S}
    x = zeros(T, d)
    ## T^{-1}(x_1,...,x_k) functions, z=x_k
    f(k) = begin
        if k==1
            z->sampler.integrals[k](T[z]) - u[sampler.variable_ordering[k]]
        else
            z->(sampler.integrals[k]([x[1:k-1]; z])/sampler.margins[k-1](x[1:k-1])) - u[sampler.variable_ordering[k]]
        end
    end
    if S<:OMF 
        for k=1:d
            x[sampler.variable_ordering[k]] = find_zero(f(k), zero(T))
        end
    else
        for k=1:d
            left, right = domain_interval(sampler.model, sampler.variable_ordering[k])
            x[sampler.variable_ordering[k]] = find_zero(f(k), (left, right))
        end
    end
    return x
end


function pullback(sampler::PSDModelSampler{d, T}, 
                        x::PSDdata{T}) where {d, T<:Number}
    f(k) = begin
        if k==1
            z->sampler.integrals[k](T[z])
        else
            z->(sampler.integrals[k]([x[1:k-1]; z])/sampler.margins[k-1](x[1:k-1]))
        end
    end
    u = similar(x)
    for k=1:d
        u[sampler.variable_ordering[k]] = f(k)(x[sampler.variable_ordering[k]])
    end
    return u
end

