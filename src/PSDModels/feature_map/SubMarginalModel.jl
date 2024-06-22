
struct PSDOrthonormalSubModel{d, d2, T} <: PSDModelOrthonormal{d, T, Nothing}
    a::PSDModelOrthonormal{d2, T}
    B::SubArray{T, 2}
    M::Symmetric{Bool, <:AbstractMatrix{Bool}}
    P::AbstractMatrix{T}
    Φ::TensorFunction{d, T}
end


function marginal_model(a::PSDModelOrthonormal{d, T}, 
    dims::Vector{Int}) where {d, T<:Number}
    P = _calculate_projector_P(a.Φ, dims)
    M = _calculate_marginal_M(a, dims)
    new_Φ = reduce_dim(a.Φ, dims)
    B = @view a.B[1:end, 1:end]
    PSDOrthonormalSubModel{d-length(dims), d, T}(a, B, M, P, new_Φ)
end

function Φ(a::PSDOrthonormalSubModel{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number}
    return a.P' * a.Φ(x)
end

function (a::PSDOrthonormalSubModel{<:Any, <:Any, T})(x::PSDdata{T}) where {T<:Number}
    v = Φ(a, x)
    return dot(v, a.a.B .* a.M, v)::T
end

# define this for Zygote, ForwardDiff, etc.
function (a::PSDOrthonormalSubModel)(x::PSDdata{T}) where {T<:Number}
    v = Φ(a, x)
    return dot(v, a.a.B .* a.M, v)::T
end

function (a::PSDOrthonormalSubModel{<:Any, <:Any, T})(x::PSDdata{T}, B::AbstractMatrix{T}) where {T<:Number}
    v = Φ(a, x)
    return dot(v, B .* a.M, v)::T
end
# define this for Zygote, ForwardDiff, etc.
function (a::PSDOrthonormalSubModel{<:Any, <:Any, T})(x::PSDdata{T}, B::AbstractMatrix) where {T<:Number}
    v = Φ(a, x)
    return dot(v, B .* a.M , v)
end

function set_coefficients!(a::PSDOrthonormalSubModel{<:Any, <:Any, T}, B::Hermitian{T}) where {T<:Number}
    throw(error("You can not set the coefficients of a submodel."))
end
