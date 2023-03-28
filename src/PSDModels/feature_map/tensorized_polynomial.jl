"""
A PSD model where the feature map is made out of orthonormal 
polynomial functions such as Chebyshev polynomials. There is an own 
implementation in order to provide orthonormal polynomials,
optimal weighted sampling, closed form derivatives and integration, etc.
For orthogonal polynomials, the package ApproxFun is used.
"""
struct PSDModelFMTensorPolynomial{T<:Number} <: AbstractPSDModelFMPolynomial{T}
    B::Hermitian{T, Matrix{T}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    Φ::Function
    function PSDModelFMTensorPolynomial(B::Hermitian{T, Matrix{T}},
                                    Φ::Function
                    ) where {T<:Number}
        new{T}(B, Φ)
    end
    function PSDModelFMTensorPolynomial{T}(B::Hermitian{T, Matrix{T}},
                                        Φ::Function
                        )where {T<:Number}
        new{T}(B, Φ)
    end
end

@inline _of_same_PSD(a::PSDModelFMTensorPolynomial{T}, B::AbstractMatrix{T}) where {T<:Number} =
                PSDModelFMTensorPolynomial(Hermitian(B), a.Φ)

"""
Marginalize the model along a given dimension according to the measure to which
the polynomials are orthogonal to.
"""
function marginalize_orth_measure(a::PSDModelFMTensorPolynomial{T}, dim::Int;
                                  measure_scale::T=1.0) where {T<:Number}
    d = dimensions(a.Φ)
    @assert 1 ≤ dim ≤ d
    M = spzeros(T, size(a.B))
    @inline δ(i::Int, j::Int) = i == j ? 1 : 0
    @inline comp_ind(x,y) = mapreduce(k->k<dim ? δ(x[k], y[k]) : δ(x[k], y[k+1]), *, 1:(d-1))
    for i=1:size(a.B, 1)
        for j=1:size(a.B, 2)
            M[i, j] = measure_scale * δ(σ_inv(a.Φ, i)[dim], σ_inv(a.Φ, j)[dim])
        end
    end
    if d-1 == 0  ## no dimension left
        return tr(M.*a.B)
    end
    new_Φ = reduce_dim(a.Φ, dim)
    P = spzeros(T, new_Φ.N, a.Φ.N)
    for i=1:new_Φ.N
        for j=1:a.Φ.N
            P[i, j] = comp_ind(σ_inv(new_Φ, i), σ_inv(a.Φ, j))
        end
    end

    B = P * (M .* a.B) * P'
    return PSDModelFMTensorPolynomial(Hermitian(Matrix(B)), new_Φ)
end

marginalize(a::PSDModelFMTensorPolynomial{T}, dim::Int) where {T<:Number} = marginalize(a, dim, x->1.0)
function marginalize(a::PSDModelFMTensorPolynomial{T}, dim::Int,
                     measure::Function) where {T<:Number}
    d = dimensions(a.Φ)
    @assert 1 ≤ dim ≤ d

    M = calculate_M_quadrature(a.Φ, dim, measure)
    if d-1 == 0  ## no dimension left
        return tr(M.*a.B)
    end

    @inline δ(i::Int, j::Int) = i == j ? 1 : 0
    @inline comp_ind(x,y) = mapreduce(k->k<dim ? δ(x[k], y[k]) : δ(x[k], y[k+1]), *, 1:(d-1))
    new_Φ = reduce_dim(a.Φ, dim)
    P = spzeros(T, new_Φ.N, a.Φ.N)
    for i=1:new_Φ.N
        for j=1:a.Φ.N
            P[i, j] = comp_ind(σ_inv(new_Φ, i), σ_inv(a.Φ, j))
        end
    end

    B = P * (M .* a.B) * P'
    return PSDModelFMTensorPolynomial(Hermitian(Matrix(B)), new_Φ)
end


function integral(a::PSDModelFMTensorPolynomial{T}, dim::Int) where {T<:Number}
    d = dimensions(a.Φ)
    @assert 1 ≤ dim ≤ d
    M = SquaredPolynomialMatrix(a.Φ, Int[dim])
    return TensorizedPolynomialTraceModel(a.B, M)
end
