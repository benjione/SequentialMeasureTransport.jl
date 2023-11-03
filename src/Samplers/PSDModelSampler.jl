using Roots: find_zero

struct PSDModelSampler{d, T<:Number, S, R, dC} <: ConditionalSampler{d, T, R, dC}
    model::PSDModelOrthonormal{d, T, S}                 # model to sample from
    margins::Vector{<:PSDModelOrthonormal{<:Any, T, S}} # start with x_{≤1}, then x_{≤2}, ...
    integrals::Vector{<:OrthonormalTraceModel{T, S}}    # integrals of marginals
    variable_ordering::AbstractVector{Int}              # variable ordering for the model
    R_map::R                                            # reference map from reference distribution to uniform
    function PSDModelSampler(model::PSDModelOrthonormal{d, T, S}, 
                             variable_ordering::AbstractVector{Int},
                             amount_cond_variable::Int) where {d, T<:Number, S}
        @assert amount_cond_variable < d
        # check that the last {amount_cond_variable} variables are the last ones in the ordering
        @assert issetequal(variable_ordering[(d-amount_cond_variable+1):d], (d-amount_cond_variable+1):d)
        model = normalize(model) # create normalized copy
        perm_model = permute_indices(model, variable_ordering) # permute dimensions
        margins = PSDModelOrthonormal{<:Any, T, S}[marginalize(perm_model, collect(k:d)) for k in 2:d]
        margins = [margins; perm_model] # add the full model at last
        integrals = map((x,k)->integral(x, k), margins, 1:d)
        new{d, T, S, Nothing, amount_cond_variable}(model, margins, integrals, variable_ordering, nothing)
    end
    function PSDModelSampler(model::PSDModelOrthonormal{d, T, S}, 
                             variable_ordering::AbstractVector{Int}) where {d, T<:Number, S}
        PSDModelSampler(model, variable_ordering, 0)
    end
end

Sampler(model::PSDModelOrthonormal{d}) where {d} = PSDModelSampler(model, Random.shuffle!(collect(1:d)))
ConditionalSampler(model::PSDModelOrthonormal{d}, 
                    amount_cond_variable::Int) where {d} = 
                        PSDModelSampler(model, [Random.shuffle!(collect(1:(d-amount_cond_variable))); 
                                Random.shuffle!(collect((d-amount_cond_variable+1):d))], amount_cond_variable)

## Pretty printing
function Base.show(io::IO, sampler::PSDModelSampler{d, T, S, R}) where {d, T, S, R}
    println(io, "PSDModelSampler{d=$d, T=$T, S=$S, R=$R}")
    println(io, "   model: ", sampler.model)
    println(io, "   order of variables: ", sampler.variable_ordering)
end

## Methods of PSDModelSampler itself


function _pushforward_first_n(sampler::PSDModelSampler{d, T, S}, 
                     u::PSDdata{T}, n::Int) where {d, T<:Number, S}
    x = zeros(T, n)
    u = @view u[sampler.variable_ordering[1:n]]
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
        for k=1:n
            x[k] = find_zero(f(k), zero(T))
        end
    else
        for k=1:n
            left, right = domain_interval(sampler.model, sampler.variable_ordering[k])
            x[k] = find_zero(f(k), (left, right))
        end
    end
    return invpermute!(x, sampler.variable_ordering[1:n])
end


function _pullback_first_n(sampler::PSDModelSampler{d, T}, 
                        x::PSDdata{T},
                        n::Int) where {d, T<:Number}
    x = @view x[sampler.variable_ordering[1:n]]
    f(k) = begin
        if k==1
            z->sampler.integrals[k](T[z])
        else
            z->(sampler.integrals[k]([x[1:k-1]; z])/sampler.margins[k-1](x[1:k-1]))
        end
    end
    u = zeros(T, n)
    for k=1:n
        u[sampler.variable_ordering[k]] = f(k)(x[k])
    end
    return u
end


## Methods for satisfying Sampler interface

function Distributions.pdf(
        sar::PSDModelSampler{d, T},
        x::PSDdata{T}
    ) where {d, T<:Number}
    return sar.model(x)
end

@inline pushforward(sampler::PSDModelSampler{d, T, S}, 
                    u::PSDdata{T}) where {d, T<:Number, S} = return _pushforward_first_n(sampler, u, d)

@inline pullback(sampler::PSDModelSampler{d, T}, 
                 x::PSDdata{T}) where {d, T<:Number} = return _pullback_first_n(sampler, x, d)


## Methods for satisfying ConditionalSampler interface

function marg_pdf(sampler::PSDModelSampler{d, T, S, R, dC}, x::PSDdata{T}) where {d, T<:Number, S, R, dC}
    dx = d-dC
    @assert length(x) == dx
    return sampler.margins[dx](x)
end

@inline marg_pushforward(sampler::PSDModelSampler{d, T, S, R, dC}, 
                         u::PSDdata{T}) where {d, T<:Number, S, R, dC} = 
                            return _pushforward_first_n(sampler, u, d-dC)
@inline marg_pullback(sampler::PSDModelSampler{d, T, S, R, dC}, 
                      x::PSDdata{T}) where {d, T<:Number, S, R, dC} = 
                            return _pullback_first_n(sampler, x, d-dC)

function cond_pushforward(sampler::PSDModelSampler{d, T, S, R, dC}, u::PSDdata{T}, x::PSDdata{T}) where {d, T<:Number, S, R, dC}
    dx = d-dC
    # @assert length(u) == dC
    # @assert length(x) == dx
    y = zeros(T, dC)
    x = @view x[sampler.variable_ordering[1:dx]]
    f(k) = begin
        z->(sampler.integrals[k+dx]([x; y[1:k-1]; z])/
                sampler.margins[k+dx-1]([x; y[1:k-1]])) - u[k]
    end
    if S<:OMF
        for k=1:d
            y[k] = find_zero(f(k), zero(T))
        end
    else
        for k=1:dC
            left, right = domain_interval(sampler.model, sampler.variable_ordering[dx+k])
            y[k] = find_zero(f(k), (left, right))
        end
    end
    return invpermute!(y, sampler.variable_ordering[dx+1:end].-dx) # might be wrong, subtract dx from variable_ordering
end


function cond_pullback(sampler::PSDModelSampler{d, T, S, R, dC}, 
                        y::PSDdata{T},
                        x::PSDdata{T}) where {d, T<:Number, S, R, dC}
    dx = d-dC
    x = @view x[sampler.variable_ordering]
    f(k) = begin
        z->(sampler.integrals[dx+k]([x; y[1:k-1]; z])/sampler.margins[dx+k-1]([x; y[1:k-1]]))
    end
    u = zeros(T, dC)
    for k=1:dC
        u[sampler.variable_ordering[k]-dx] = f(k)(y[k])
    end
    return u
end
