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
        perm_model = permute_indices(model, variable_ordering) # permute dimensions
        margins = PSDModelOrthonormal{<:Any, T, S}[marginalize(perm_model, collect(k:d)) for k in 2:d]
        margins = [margins; perm_model] # add the full model at last
        integrals = map((x,k)->integral(x, k), margins, 1:d)
        new{d, T, S, Nothing}(model, margins, integrals, variable_ordering, nothing)
    end
end

Sampler(model::PSDModelOrthonormal{d}) where {d} = PSDModelSampler(model, Random.shuffle!(collect(1:d)))

## Pretty printing
function Base.show(io::IO, sampler::PSDModelSampler{d, T, S, R}) where {d, T, S, R}
    println(io, "PSDModelSampler{d=$d, T=$T, S=$S, R=$R}")
    println(io, "   model: ", sampler.model)
    println(io, "   order of variables: ", sampler.variable_ordering)
end

function Distributions.pdf(
        sar::PSDModelSampler{d, T},
        x::PSDdata{T}
    ) where {d, T<:Number}
    return sar.model(x)
end


function pushforward(sampler::PSDModelSampler{d, T, S}, u::PSDdata{T}) where {d, T<:Number, S}
    x = zeros(T, d)
    u = @view u[sampler.variable_ordering]
    ## T^{-1}(x_1,...,x_k) functions, z=x_k
    f(k) = begin
        if k==1
            z->sampler.integrals[k](T[z]) - u[k] #u[sampler.variable_ordering[k]]
        else
            z->(sampler.integrals[k]([x[1:k-1]; z])/
                    sampler.margins[k-1](x[1:k-1])) - u[k] #u[sampler.variable_ordering[k]]
        end
    end
    if S<:OMF 
        for k=1:d
            x[k] = find_zero(f(k), zero(T))
        end
    else
        for k=1:d
            left, right = domain_interval(sampler.model, sampler.variable_ordering[k])
            x[k] = find_zero(f(k), (left, right))
        end
    end
    return invpermute!(x, sampler.variable_ordering)
end


function pullback(sampler::PSDModelSampler{d, T}, 
                        x::PSDdata{T}) where {d, T<:Number}
    x = @view x[sampler.variable_ordering]
    f(k) = begin
        if k==1
            z->sampler.integrals[k](T[z])
        else
            z->(sampler.integrals[k]([x[1:k-1]; z])/sampler.margins[k-1](x[1:k-1]))
        end
    end
    u = similar(x)
    for k=1:d
        u[sampler.variable_ordering[k]] = f(k)(x[k])
    end
    return u
end

