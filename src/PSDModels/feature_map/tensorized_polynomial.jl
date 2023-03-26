using ApproxFun
using SparseArrays

include("../../functions/TensorPolynomial.jl")

"""
A PSD model where the feature map is made out of orthonormal 
polynomial functions such as Chebyshev polynomials. There is an own 
implementation in order to provide orthonormal polynomials,
optimal weighted sampling, closed form derivatives and integration, etc.
For orthogonal polynomials, the package ApproxFun is used.
"""
struct PSDModelFMTensorPolynomial{T<:Number} <: AbstractPSDModelFMPolynomial{T}
    B::Hermitian{Float64, Matrix{Float64}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    Φ::Function
    function PSDModelFMTensorPolynomial{T}(B::Hermitian{Float64, Matrix{Float64}},
                                    Φ::Function
                    ) where {T<:Number}
        new{T}(B, Φ)
    end
end

@inline _of_same_PSD(a::PSDModelFMTensorPolynomial{T}, B::AbstractMatrix{T}) where {T<:Number} =
                PSDModelFMPolynomial{T}(Hermitian(B), a.Φ)

"""
Marginalize the model along a given dimension according to the measure to which
the polynomials are orthogonal to.
"""
function marginalize_orth_measure(a::PSDModelFMTensorPolynomial{T}, dim::Int) where {T<:Number}
    d = dimensions(a.Φ)
    @assert 1 ≤ dim ≤ d
    M = spzeros(T, size(a.B))
    @inline δ(i::Int, j::Int) = i == j ? 1 : 0
    @inline comp_ind(x,y) = mapreduce(k->k<dim ? δ(x[k], y[k]) : δ(x[k], y[k+1]), *, 1:(d-1))
    for i=1:size(a.B, 1)
        for j=1:size(a.B, 2)
            M[i, j] = δ(σ_inv(a.Φ, i)[dim], σ_inv(a.Φ, j)[dim])
        end
    end
    new_Φ = reduce_dim(a.Φ, dim)
    P = spzeros(T, new_Φ.N, a.Φ.N)
    for i=1:new_Φ.N
        for j=1:a.Φ.N
            P[i, j] = comp_ind(σ_inv(new_Φ, i), σ_inv(a.Φ, j))
        end
    end
    B = P * (M .* a.B) * P'
    return PSDModelFMTensorPolynomial{T}(Hermitian(Matrix(B)), new_Φ)
end


function integral(a::PSDModelFMTensorPolynomial{T}, dim::Int) where {T<:Number}
    d = dimensions(a.Φ)
    @assert 1 ≤ dim ≤ d
    sp = a.Φ.space.spaces[dim]

    B = P * B_tilde
    # model not PSD anymore, use scalar model for evaluation
    return ScalarModel{T}(Matrix(B), Φ_new)
end
