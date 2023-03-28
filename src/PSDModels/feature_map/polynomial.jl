"""
A PSD model where the feature map is made out of polynomial functions such as
Chebyshev polynomials. There is an own implementation in order to provide
optimal weighted sampling, closed form derivatives and integration, etc.
As a polynomial package, ApproxFun is used.
"""
struct PSDModelFMPolynomial{T<:Number} <: AbstractPSDModelFMPolynomial{T}
    B::Hermitian{Float64, Matrix{Float64}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    Π::Space                                # Polynomial space of the feature map
    func_vec::Vector{Fun}                   # Vector of functions that make up the feature map
    function PSDModelFMPolynomial{T}(B::Hermitian{Float64, Matrix{Float64}}, 
                                    Π::Space,
                                    func_vec::Vector{<:Fun}
                    ) where {T<:Number}
        @assert length(func_vec) == size(B, 1)
        new{T}(B, Π, func_vec)
    end
end

function PSDModelFMPolynomial{T}(B::Hermitian{Float64, Matrix{Float64}}, 
                Π::Space
            ) where {T<:Number}
    N = size(B, 1)
    func_vec = Fun[Fun(Π, Float64[zeros(d); 1.0]) for d=0:(N-1)]
    PSDModelFMPolynomial{T}(B, Π, func_vec)
end

@inline _of_same_PSD(a::PSDModelFMPolynomial{T}, B::AbstractMatrix{T}) where {T<:Number} =
                PSDModelFMPolynomial{T}(Hermitian(B), a.Π, a.func_vec)

"""
Φ(a::PSDModelFMPolynomial, x::PSDdata{T}) where {T<:Number}

Returns the feature map of the PSD model at x.
"""
function Φ(a::PSDModelFMPolynomial{T}, x::PSDdata{T}) where {T<:Number}
    return map(f-> f(x), a.func_vec)
end

function integral(a::PSDModelFMPolynomial{T}, dim::AbstractVector) where {T<:Number}
    return integral(_to_FMMat(a), dim)
end

function _to_FMMat(a::PSDModelFMPolynomial{T}) where {T<:Number}
    func_mat = Fun[f1 * f2 for f1 in a.func_vec, f2 in a.func_vec]
    Π = func_mat[1, 1].space
    return PolynomialTraceModel{T}(copy(a.B), a.Π, func_mat)
end
