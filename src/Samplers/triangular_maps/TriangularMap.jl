


abstract type AbstractTriangularMap{d, dC, T} <: ConditionalMapping{d,dC,T} end

"""
    TriangularMap{d, dC, T}
    A generic implementation of triangular transport maps. In practice, this is
    overriten by a specific implementation of a triangular transport map.
    The functions defined here can be reused, by defining the `map[k]` and `jacobian[k]`

    The map is defned as Q(x)_k = Q(x_k | x_{<k}) and the jacobian as ∂_k Q(x)_k = ∂_k Q(x_k | x_{<k})
    where x_{<k} = [x_1, ..., x_{k-1}]
"""
struct TriangularMap{d, dC, T<:Number} <: ConditionalMapping{d,dC,T}
    MonotoneMap::Vector{Function}                  # Q(x_k | x_{<k})
    jacobian::Vector{Function}             # ∂_k Q(x_k | x_{<k})
    variable_ordering::Vector{Int}          # variable ordering for the model
    function TriangularSampler(map::Vector{Function}, 
            jacobian::Vector{Function},
                            variable_ordering::Vector{Int}, dC::Int)
        d = length(map)
        @assert dC < d
        # check that the last {dC} variables are the last ones in the ordering
        @assert issetequal(variable_ordering[(d-dC+1):d], (d-dC+1):d)
        
        new{d, dC, T}(map, jacobian, variable_ordering)
    end
    function TriangularSampler(map::Vector{Function}, jacobian::Vector{Function}, variable_ordering::Vector{Int})
        TriangularSampler(map, jacobian, variable_ordering, 0)
    end
end

@inline MonotoneMap(sampler::TriangularMap{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = sampler.MonotoneMap[k](x[1:k])
@inline ∂k_MonotoneMap(sampler::TriangularMap{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = sampler.jacobian[k](x[1:k])

function _pushforward_first_n(sampler::AbstractTriangularMap{d, <:Any, T}, 
                     x::PSDdata{T}, n::Int) where {d, T<:Number}
    x = @view x[sampler.variable_ordering[1:n]]
    u = Vector{T}(undef, n)
    for k=1:n
        u[sampler.variable_ordering[k]] = MonotoneMap(sampler, x[1:k], k)
    end
    return u::Vector{T}
end


function _pullback_first_n(sampler::AbstractTriangularMap{d, <:Any, T}, 
                    u::PSDdata{T},
                    n::Int) where {d, T<:Number}
    x = Vector{T}(undef, n)
    u = @view u[sampler.variable_ordering[1:n]]
    for k=1:n
        func(z) = begin
            @inbounds x[k] = z
            return MonotoneMap(sampler, x[1:k], k) - @inbounds u[k]
        end
        @inbounds x[k] = find_zero(func, zero(T))
    end
    return invpermute!(x, sampler.variable_ordering[1:n])
end


@inline pushforward(sampler::AbstractTriangularMap{d, <:Any, T}, 
                    u::PSDdata{T}) where {d, T<:Number} = return _pushforward_first_n(sampler, u, d)

@inline pullback(sampler::AbstractTriangularMap{d, <:Any, T}, 
                 x::PSDdata{T}) where {d, T<:Number} = return _pullback_first_n(sampler, x, d)

@inline Jacobian(sampler::AbstractTriangularMap{d, <:Any, T}, 
                 x::PSDdata{T}
        ) where {d, T<:Number} = prod([∂k_MonotoneMap(sampler, x[1:k], k) for k=1:d])
@inline inverse_Jacobian(sampler::AbstractTriangularMap{d, <:Any, T}, 
                        x::PSDdata{T}
        ) where {d, T<:Number} = 1/Jacobian(sampler, pullback(sampler, x))


## Methods for satisfying ConditionalSampler interface

function marginal_Jacobian(sampler::AbstractTriangularMap{d, dC, T}, x::PSDdata{T}) where {d, dC, T<:Number}
    return prod([∂k_MonotoneMap(sampler, x[1:k], k) for k=1:(d-dC)])
end

function marginal_inverse_Jacobian(sampler::AbstractTriangularMap{d, dC, T}, x::PSDdata{T}) where {d, dC, T<:Number}
    return 1/marginal_Jacobian(sampler, marginal_pullback(sampler, x))
end

@inline marginal_pushforward(sampler::AbstractTriangularMap{d, dC, T}, 
                         u::PSDdata{T}) where {d, T<:Number, dC} = 
                            return _pushforward_first_n(sampler, u, d-dC)
@inline marginal_pullback(sampler::AbstractTriangularMap{d, dC, T}, 
                      x::PSDdata{T}) where {d, T<:Number, dC} = 
                            return _pullback_first_n(sampler, x, d-dC)

function conditional_pushforward(sampler::AbstractTriangularMap{d, dC, T}, y::PSDdata{T}, x::PSDdata{T}) where {d, dC, T<:Number}
    dx = d-dC
    x = @view x[sampler.variable_ordering]
    u = zeros(T, dC)
    for k=1:dC
        u[sampler.variable_ordering[k]-dx] = MonotoneMap(sampler, [x; y[1:k]], dx+k)
    end
    return u
end


function conditional_pullback(sampler::AbstractTriangularMap{d, dC, T}, 
                        u::PSDdata{T},
                        x::PSDdata{T}) where {d, T<:Number, dC}
    dx = d-dC
    y = zeros(T, dC)
    x = @view x[sampler.variable_ordering[1:dx]]
    for k=1:dC
        y[k] = find_zero(z->MonotoneMap(sampler, [x; y[1:k-1]; z], dx+k) - u[k], zero(T))
    end
    return invpermute!(y, sampler.variable_ordering[dx+1:end].-dx)
end

