

struct SqPolynomial{d, T}
    f_map::TensorFunction{d, T}
    coeff::Vector{T}
end


function SqPolynomial(sp::Space, order::Int)
    p = downwardClosed_Polynomial(Float64, 
                sp, order)
    coeffs = zeros(Float64, p.N)
    return SqPolynomial{length(sp.spaces), Float64}(p, coeffs)
end


function (a::SqPolynomial{<:Any, T})(x::PSDdata{T}) where {T <: Number}
    return dot(a.f_map(x), a.coeff)^2
end


function _Gramm_matrix(p::TensorFunction{<:Any, T}, X::PSDDataVector{T}, 
                weights::AbstractVector{T}) where {T <: Number}
    V = p.(X)
    1/length(weights) * sum(v * v' * w for (v, w) in zip(V, weights))
end


function _b_vector(p::TensorFunction{<:Any, T}, X::PSDDataVector{T}, 
        Y::AbstractVector{T}, weights::AbstractVector{T}) where {T <: Number}
    (1.0/length(weights)) * sum(p.(X) .* Y .* weights)
end

function _sq_poly_Hellinger_fit!(a::SqPolynomial{d, T}, X::PSDDataVector{T}, 
                        Y::AbstractVector{T}, weights::AbstractVector{T}) where {d, T<:Number}
    _Y = sqrt.(Y)
    G = _Gramm_matrix(a.f_map, X, weights)
    b = _b_vector(a.f_map, X, _Y, weights)
    coeffs = G \ b
    a.coeff .= coeffs
    return a
end


function _to_PSD(a::SqPolynomial{d, T}) where {d, T}
    B = Hermitian(a.coeff * a.coeff')
    return PSDModelPolynomial{T}(B, a.f_map)
end