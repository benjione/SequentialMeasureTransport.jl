abstract type PSDModelOrthonormal{d, T, S} <: AbstractPSDModelFM{T} end

dimension(a::PSDModelOrthonormal{d}) where {d} = d

## special feature map models:
include("polynomial.jl")
include("SubMarginalModel.jl")


"""
Φ(a::PSDModelFM, x::PSDdata{T}) where {T<:Number}

Returns the feature map of the PSD model at x.
"""
@inline function Φ(a::PSDModelOrthonormal{<:Any, T, S}, x::PSDdata{T}) where {T<:Number, S<:OMF{T}}
    return a.Φ(ξ(a.mapping, x)) * sqrt(1/x_deriv_prod(a.mapping, ξ(a.mapping, x)))
end

@inline function Φ(a::PSDModelOrthonormal{<:Any, T, S}, x::PSDdata{T}) where {T<:Number, S<:ConditionalMapping{<:Any, <:Any, T}}
    return a.Φ(pullback(a.mapping, x)) * sqrt(inverse_Jacobian(a.mapping, x))
end

# faster evaluation for mapping, since pushforward of function faster than doing manually
function (a::PSDModelOrthonormal{<:Any, T, S})(x::PSDdata{T}) where {T<:Number, S<:ConditionalMapping{<:Any, <:Any, T}}
    _help(x) = begin
        v = a.Φ(x)
        return dot(v, a.B, v)
    end
    return pushforward(a.mapping, _help)(x)
end


domain_interval_left(a::PSDModelOrthonormal{<:Any, <:Any, <:OMF}, k::Int) = -∞
domain_interval_right(a::PSDModelOrthonormal{<:Any, <:Any, <:OMF}, k::Int) = +∞


domain_interval_left(a::PSDModelOrthonormal, k::Int) = domain_interval(a, k)[1]
domain_interval_right(a::PSDModelOrthonormal, k::Int) = domain_interval(a, k)[2]
domain_interval_left(a::PSDModelOrthonormal{d}) where {d} = domain_interval_left.(Ref(a), collect(1:d))
domain_interval_right(a::PSDModelOrthonormal{d}) where {d} = domain_interval_right.(Ref(a), collect(1:d))

_volume(a::PSDModelOrthonormal{<:Any, <:Any, <:OMF}) = throw(error("Not implemented!"))
_volume(a::PSDModelOrthonormal) = prod(domain_interval_right(a) - domain_interval_left(a))

## general interface
_tensorizer(a::PSDModelOrthonormal) = throw(error("Not implemented!"))
_remove_mapping(a::PSDModelOrthonormal{<:Any, <:Any, <:ConditionalMapping}) = throw(error("Not implemented!"))

## for greedy downward closed approximation
next_index_proposals(a::PSDModelOrthonormal) = next_index_proposals(_tensorizer(a))
create_proposal(a::PSDModelOrthonormal, index::Vector{Int}) = throw(error("Not implemented!"))

function permute_indices(a::PSDModelOrthonormal, perm::Vector{Int})
    return _of_same_PSD(a, a.B, permute_indices(a.Φ, perm))
end

## Interface between OMF and non OMF mapping
function x(a::PSDModelOrthonormal{<:Any, <:Any, <:OMF}, ξ::PSDdata{T}) where {T<:Number}
    return x(a.mapping, ξ)
end
function x(a::PSDModelOrthonormal{<:Any, <:Any, <:Nothing}, x::PSDdata{T}) where {T<:Number}
    L = domain_interval_left(a)
    R = domain_interval_right(a)
    return 2.0 * (x .- L) ./ (R .- L) .- 1.0
end

function ξ(a::PSDModelOrthonormal{<:Any, <:Any, <:OMF}, x::PSDdata{T}) where {T<:Number}
    return ξ(a.mapping, x)
end
function ξ(a::PSDModelOrthonormal{<:Any, <:Any, Nothing}, x::PSDdata{T}) where {T<:Number}
    L = domain_interval_left(a)
    R = domain_interval_right(a)
    return ((x .+ 1.0) ./ 2.0) .* (R .- L) .+ L
end


function _calculate_marginal_M(a::PSDModelOrthonormal{d, T}, dim::Int) where {d, T<:Number}
    @assert 1 ≤ dim ≤ d
    M = spzeros(T, size(a.B))
    @inline δ(i::Int, j::Int) = i == j ? 1 : 0
    for i=1:size(a.B, 1)
        for j=i:size(a.B, 2)
            M[i, j] = δ(σ_inv(a.Φ, i)[dim], σ_inv(a.Φ, j)[dim])
        end
    end
    return Symmetric(M)
end

function _calculate_marginal_M(a::PSDModelOrthonormal{d, T}, dim::Vector{Int}) where {d, T<:Number}
    M = ones(Bool, size(a.B))
    for _d in dim
        M .= M .&& _calculate_marginal_M(a, _d)
    end
    return Symmetric(M)
end

function _calculate_projector_P(dim::Int, 
                new_Φ::TensorFunction{<:Any, T}, 
                old_Φ::TensorFunction{d, T}
            ) where {d, T<:Number}
    @assert 1 ≤ dim ≤ d
    @inline δ(i::Int, j::Int) = i == j ? 1 : 0
    @inline comp_ind(x,y) = mapreduce(k->k<dim ? δ(x[k], y[k]) : δ(x[k], y[k+1]), *, 1:(d-1))
    P = spzeros(T, new_Φ.N, old_Φ.N)
    for i=1:new_Φ.N
        for j=1:old_Φ.N
            P[i, j] = comp_ind(σ_inv(new_Φ, i), σ_inv(old_Φ, j))
        end
    end
    return P
end

function _calculate_projector_P(Φ::TensorFunction, dims::Vector{Int})
    _dims = sort(dims, rev=true)
    new_Φ = reduce_dim(Φ, dims[1])
    P = nothing
    for dim in _dims
        new_Φ = reduce_dim(Φ, dim)
        if P === nothing
            P = _calculate_projector_P(dim, new_Φ, Φ)
        else
            P = _calculate_projector_P(dim, new_Φ, Φ) * P
        end 
        Φ = new_Φ
    end
    return P
end


"""
Marginalize the model along a given dimension according to the measure to which
the polynomials are orthogonal to.
"""
function marginalize_orth_measure(a::PSDModelOrthonormal{d, T}, dim::Int;
                                  measure_scale::T=1.0) where {d, T<:Number}
    M = _calculate_marginal_M(a, dim) * measure_scale
    if d-1 == 0  ## no dimension left
        return tr(M.*a.B)
    end
    new_Φ = reduce_dim(a.Φ, dim)
    @inline comp_ind(x,y) = mapreduce(k->k<dim ? δ(x[k], y[k]) : δ(x[k], y[k+1]), *, 1:(d-1))
    P = _calculate_projector_P(dim, new_Φ, a.Φ)

    B = P * (M .* a.B) * P'
    return _of_same_PSD(a, Hermitian(Matrix(B)), new_Φ)
end

function marginalize_orth_measure(a::PSDModelOrthonormal{d, T}, 
                        dims::Vector{Int}) where {d, T<:Number}
    @assert 1 ≤ minimum(dims) ≤ maximum(dims) ≤ d
    dims = sort(dims)
    for dim in reverse(dims) ## reverse order to avoid changing the indices
        a = marginalize_orth_measure(a, dim)
    end
    return a
end
