using Roots: find_zero

struct PSDModelSampler{d, dC, T<:Number, S} <: AbstractCondSampler{d, dC, T, Nothing, Nothing}
    perm_model::PSDModelOrthonormal{d, T, S}                 # permuted model to sample from
    margins::Vector{<:PSDModelOrthonormal{<:Any, T, S}} # start with x_{≤1}, then x_{≤2}, ...
    # margins::Vector{Function}                           # start with x_{≤1}, then x_{≤2}, ...
    integrals::Vector{<:OrthonormalTraceModel{T, S}}    # integrals of marginals
    # integrals::Vector{Function}                         # integrals of marginals
    variable_ordering::Vector{Int}                      # variable ordering for the model
    function PSDModelSampler(model::PSDModelOrthonormal{d, T, S}, 
                             variable_ordering::Vector{Int},
                             dC::Int) where {d, T<:Number, S}
        @assert dC < d
        # check that the last {dC} variables are the last ones in the ordering
        @assert issetequal(variable_ordering[(d-dC+1):d], (d-dC+1):d)
        model = normalize(model) # create normalized copy
        perm_model = permute_indices(model, variable_ordering) # permute dimensions
        margins = PSDModelOrthonormal{<:Any, T, S}[marginal_model(perm_model, collect(k:d)) for k in 2:d]
        margins = [margins; perm_model] # add the full model at last
        real_margins = [marginalize(perm_model, collect(k:d)) for k in 2:d]
        real_margins = [real_margins; perm_model]
        integrals = map((x,k)->integral(x, k), real_margins, 1:d)
        new{d, dC, T, S}(perm_model, margins, integrals, variable_ordering)
    end
    function PSDModelSampler(model::PSDModelOrthonormal{d, T, S}, 
                             variable_ordering::Vector{Int}) where {d, T<:Number, S}
        PSDModelSampler(model, variable_ordering, 0)
    end
end

Sampler(model::PSDModelOrthonormal{d}) where {d} = Sampler(model, Random.shuffle!(collect(1:d)))
Sampler(model::PSDModelOrthonormal{d}, variable_ordering::Vector{Int}) where {d} = PSDModelSampler(model, variable_ordering)
ConditionalSampler(model::PSDModelOrthonormal{d}, 
                    dC::Int) where {d} = 
                    ConditionalSampler(model, dC, [Random.shuffle!(collect(1:(d-dC))); 
                                Random.shuffle!(collect((d-dC+1):d))])
ConditionalSampler(model::PSDModelOrthonormal{d}, dC::Int,
                    variable_ordering::Vector{Int}) where {d} = 
                        PSDModelSampler(model, variable_ordering, dC)


function Sampler(model::PSDModelOrthonormal{<:Any, <:Any, <:Mapping}, variable_ordering::Vector{Int})
    smp = Sampler([model.mapping], nothing)
    smp2 = Sampler(_remove_mapping(model), variable_ordering)
    add_layer!(smp, smp2)
    return smp
end
function Sampler(model::PSDModelOrthonormal{<:Any, <:Any, <:CondSampler}, variable_ordering::Vector{Int})
    smp = model.mapping
    smp2 = Sampler(_remove_mapping(model), variable_ordering)
    add_layer!(smp, smp2)
    return smp
end
function ConditionalSampler(model::PSDModelOrthonormal{<:Any, <:Any, <:ConditionalMapping}, 
                            dC::Int, variable_ordering::Vector{Int})
    smp = CondSampler([model.mapping], nothing)
    smp2 = ConditionalSampler(_remove_mapping(model), dC, variable_ordering)
    add_layer!(smp, smp2)
    return smp
end
function ConditionalSampler(model::PSDModelOrthonormal{<:Any, <:Any, <:CondSampler{<:Any, _dC}}, 
            dC::Int, variable_ordering::Vector{Int}) where {_dC}
    @assert dC == _dC
    smp = model.mapping
    smp2 = ConditionalSampler(_remove_mapping(model), dC, variable_ordering)
    add_layer!(smp, smp2)
    return smp
end

## Pretty printing
function Base.show(io::IO, sampler::PSDModelSampler{d, T, S, R}) where {d, T, S, R}
    println(io, "PSDModelSampler{d=$d, dC=$T, T=$S, S=$R}")
    println(io, "   model: ", sampler.perm_model)
    println(io, "   order of variables: ", sampler.variable_ordering)
end

## Methods of PSDModelSampler itself


function _pushforward_first_n(sampler::PSDModelSampler{d, <:Any, T, S}, 
                     u::PSDdata{T}, n::Int) where {d, T<:Number, S}
    x = zeros(T, n)
    u = @view u[sampler.variable_ordering[1:n]]
    ## T^{-1}(x_1,...,x_k) functions, z=x_k
    f(k::Int) = begin
        A = if k==d
            sampler.margins[k].B
        else 
            A = sampler.margins[k].P * (sampler.margins[k].M .* sampler.margins[k].B) * sampler.margins[k].P'
        end
        if k==1
            z->sampler.integrals[k](T[z], A) - u[k] #u[sampler.variable_ordering[k]]
        else
            z->(sampler.integrals[k]([x[1:k-1]; z], A)/
                    sampler.margins[k-1](x[1:k-1])) - u[k] #u[sampler.variable_ordering[k]]
        end
    end
    if S<:OMF 
        for k=1:n
            x[k] = find_zero(f(k), zero(T))::T
        end
    else
        for k=1:n
            left, right = domain_interval(sampler.perm_model, k)
            x[k] = find_zero(f(k), (left, right))::T
        end
    end
    return invpermute!(x, sampler.variable_ordering[1:n])
end


function _pullback_first_n(sampler::PSDModelSampler{d, <:Any, T}, 
                        x::PSDdata{T},
                        n::Int) where {d, T<:Number}
    x = @view x[sampler.variable_ordering[1:n]]
    f(k::Int) = begin
        A = if k==d
            sampler.margins[k].B
        else 
            A = sampler.margins[k].P * (sampler.margins[k].M .* sampler.margins[k].B) * sampler.margins[k].P'
        end
        if k==1
            z->sampler.integrals[k](T[z], A)
        else
            z->(sampler.integrals[k]([x[1:k-1]; z], A)/sampler.margins[k-1](x[1:k-1]))
        end
    end
    u = Vector{T}(undef, n)
    for k=1:n
        u[sampler.variable_ordering[k]] = f(k)(x[k])
    end
    # typestable function
    _rounding(u::Vector{T}) = begin
        map(u) do x
            if zero(T) ≤ x ≤ one(T)
                x
            else
                zero(T) > x ? zero(T) : one(T)
            end
        end
    end
    u = _rounding(u)
    return u::Vector{T}
end


## Methods for satisfying Sampler interface

function Distributions.pdf(
        sar::PSDModelSampler{d, <:Any, T},
        x::PSDdata{T}
    ) where {d, T<:Number}
    return sar.perm_model(x[sar.variable_ordering])::T
end

function Distributions.logpdf(
        sar::PSDModelSampler{d, <:Any, T},
        x::PSDdata{T}
    ) where {d, T<:Number}
    return log(sar.perm_model(x[sar.variable_ordering])+ϵ_log)::T
end


function marginal_pdf(sampler::PSDModelSampler{d, dC, T, S}, x::PSDdata{T}) where {d, dC, T<:Number, S}
    return sampler.margins[d-dC](x[sampler.variable_ordering[1:d-dC]])
end

marginal_logpdf(sampler::PSDModelSampler{d, dC, T, S}, x::PSDdata{T}) where {d, dC, T<:Number, S} = log(marginal_pdf(sampler, x)+ϵ_log)

@inline pushforward(sampler::PSDModelSampler{d, <:Any, T, S}, 
                    u::PSDdata{T}) where {d, T<:Number, S} = return _pushforward_first_n(sampler, u, d)

@inline pullback(sampler::PSDModelSampler{d, <:Any, T}, 
                 x::PSDdata{T}) where {d, T<:Number} = return _pullback_first_n(sampler, x, d)

@inline Jacobian(sampler::PSDModelSampler{d, <:Any, T}, 
                 x::PSDdata{T}
        ) where {d, T<:Number} = 1/inverse_Jacobian(sampler, pushforward(sampler, x))
@inline inverse_Jacobian(sampler::PSDModelSampler{d, <:Any, T}, 
                        x::PSDdata{T}
        ) where {d, T<:Number} = Distributions.pdf(sampler, x)


## Methods for satisfying ConditionalSampler interface

function marginal_Jacobian(sampler::PSDModelSampler{d, dC, T, S}, x::PSDdata{T}) where {d, dC, T<:Number, S}
    return 1/marginal_inverse_Jacobian(sampler, marginal_pushforward(sampler, x))
end

function marginal_inverse_Jacobian(sampler::PSDModelSampler{d, dC, T, S}, x::PSDdata{T}) where {d, dC, T<:Number, S}
    return marginal_pdf(sampler, x)
end

@inline marginal_pushforward(sampler::PSDModelSampler{d, dC, T, S}, 
                         u::PSDdata{T}) where {d, T<:Number, S, dC} = 
                            return _pushforward_first_n(sampler, u, d-dC)
@inline marginal_pullback(sampler::PSDModelSampler{d, dC, T, S}, 
                      x::PSDdata{T}) where {d, T<:Number, S, dC} = 
                            return _pullback_first_n(sampler, x, d-dC)

function conditional_pushforward(sampler::PSDModelSampler{d, dC, T, S}, u::PSDdata{T}, x::PSDdata{T}) where {d, T<:Number, S, dC}
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
        for k=1:dC
            y[k] = find_zero(f(k), zero(T))
        end
    else
        for k=1:dC
            left, right = domain_interval(sampler.perm_model, dx+k)
            y[k] = find_zero(f(k), (left, right))
        end
    end
    return invpermute!(y, sampler.variable_ordering[dx+1:end].-dx)
end


function conditional_pullback(sampler::PSDModelSampler{d, dC, T, S}, 
                        y::PSDdata{T},
                        x::PSDdata{T}) where {d, T<:Number, S, dC}
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
