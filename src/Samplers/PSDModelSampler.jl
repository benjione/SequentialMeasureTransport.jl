using Roots: find_zero

struct PSDModelSampler{d, T<:Number, S, R} <: Sampler{d, T, R}
    model::PSDModelOrthonormal{d, T, S} # model to sample from
    margins::Vector{<:PSDModelOrthonormal{<:Any, T, S}} # start with x_{≤1}, then x_{≤2}, ...
    integrals::Vector{<:OrthonormalTraceModel{T, S}} # integrals of marginals
    R_map::R    # reference map from reference distribution to uniform
    function PSDModelSampler(model::PSDModelOrthonormal{d, T, S}) where {d, T<:Number, S}
        model = normalize(model) # create normalized copy
        margins = [marginalize(model, collect(k:d)) for k in 2:d]
        margins = [margins; model] # add the full model as last
        integrals = map((x,k)->integral(x, k), margins, 1:d)
        new{d, T, S, Nothing}(model, margins, integrals, nothing)
    end
end

Sampler(model::PSDModelOrthonormal) = PSDModelSampler(model)

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
            z->sampler.integrals[k](z) - u[k]
        else
            z->(sampler.integrals[k]([x[1:k-1]; z])/sampler.margins[k-1](x[1:k-1])) - u[k]
        end
    end
    if S<:OMF 
        for k=1:d
            x[k] = find_zero(f(k), 0.0)
        end
    else
        for k=1:d
            left, right = domain_interval(sampler.model, k)
            x[k] = find_zero(f(k), (left, right))
        end
    end
    return x
end


function pullback(sampler::PSDModelSampler{d, T}, 
                        x::PSDdata{T}) where {d, T<:Number}
    f(k) = begin
        if k==1
            z->sampler.integrals[k](z)
        else
            z->(sampler.integrals[k]([x[1:k-1]; z])/sampler.margins[k-1](x[1:k-1]))
        end
    end
    u = similar(x)
    for k=1:d
        u[k] = f(k)(x[k])
    end
    return u
end

