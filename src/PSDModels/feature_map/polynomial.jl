"""
A PSD model where the feature map is made out of orthonormal 
polynomial functions such as Chebyshev polynomials. There is an own 
implementation in order to provide orthonormal polynomials,
optimal weighted sampling, closed form derivatives and integration, etc.
For orthogonal polynomials, the package ApproxFun is used.
"""
struct PSDModelPolynomial{d, T<:Number} <: AbstractPSDModelPolynomial{T}
    B::Hermitian{T, <:AbstractMatrix{T}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    Φ::FMTensorPolynomial{d, T}
    function PSDModelPolynomial(B::Hermitian{T, <:AbstractMatrix{T}},
                                    Φ::FMTensorPolynomial{d, T}
                    ) where {d, T<:Number}
        new{d, T}(B, Φ)
    end
    function PSDModelPolynomial{T}(B::Hermitian{T, <:AbstractMatrix{T}},
                                    Φ::FMTensorPolynomial{d, T}
                        )where {d, T<:Number}
        new{d, T}(B, Φ)
    end
end

@inline _of_same_PSD(a::PSDModelPolynomial{<:Any, T}, B::AbstractMatrix{T}) where {T<:Number} =
                PSDModelPolynomial(Hermitian(B), a.Φ)

"""
Marginalize the model along a given dimension according to the measure to which
the polynomials are orthogonal to.
"""
function marginalize_orth_measure(a::PSDModelPolynomial{d, T}, dim::Int;
                                  measure_scale::T=1.0) where {d, T<:Number}
    @assert 1 ≤ dim ≤ d
    M = spzeros(T, size(a.B))
    @inline δ(i::Int, j::Int) = i == j ? 1 : 0
    @inline comp_ind(x,y) = mapreduce(k->k<dim ? δ(x[k], y[k]) : δ(x[k], y[k+1]), *, 1:(d-1))
    for i=1:size(a.B, 1)
        for j=i:size(a.B, 2)
            M[i, j] = measure_scale * δ(σ_inv(a.Φ, i)[dim], σ_inv(a.Φ, j)[dim])
        end
    end
    M = Symmetric(M)
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
    return PSDModelPolynomial(Hermitian(Matrix(B)), new_Φ)
end

function marginalize_orth_measure(a::PSDModelPolynomial{d, T}, 
                        dims::Vector{Int}) where {d, T<:Number}
    @assert 1 ≤ minimum(dims) ≤ maximum(dims) ≤ d
    dims = sort(dims)
    for dim in reverse(dims) ## reverse order to avoid changing the indices
        a = marginalize_orth_measure(a, dim)
    end
    return a
end

marginalize(a::PSDModelPolynomial{<:Any, T}, dim::Int) where {T<:Number} = marginalize(a, dim, x->1.0)
function marginalize(a::PSDModelPolynomial{d, T}, dim::Int,
                     measure::Function) where {d, T<:Number}
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
    return PSDModelPolynomial(Hermitian(Matrix(B)), new_Φ)
end

marginalize(a::PSDModelPolynomial{<:Any, T}, dims::Vector{Int}) where {T<:Number} = marginalize(a, dims, x->1.0)
function marginalize(a::PSDModelPolynomial{d, T}, dims::Vector{Int},
                    measure::Function) where {d, T<:Number}
    @assert 1 ≤ minimum(dims) ≤ maximum(dims) ≤ d
    dims = sort(dims)
    for dim in reverse(dims) ## reverse order to avoid changing the indices
        a = marginalize(a, dim, measure)
    end
    return a
end

"""
    function integral(a::PSDModelPolynomial{d, T}, dim::Int; C=nothing)

Integrate the model along a given dimension. The integration constant C gives
the x value of where it should start. If C is not given, it is assumed to be
the beginning of the interval.
"""
function integral(a::PSDModelPolynomial{d, T}, dim::Int; C=nothing) where {d, T<:Number}
    @assert 1 ≤ dim ≤ d
    if C === nothing
        C = leftendpoint(p.space.spaces[dim].domain)
    end
    M = SquaredPolynomialMatrix(a.Φ, Int[dim]; C=C)
    return PolynomialTraceModel(a.B, M)
end

normalize_orth_measure!(a::PSDModelPolynomial{<:Any, T}) where {T<:Number} = a.B .= a.B * (1/tr(a.B))
normalize!(a::PSDModelPolynomial{d, T}) where {d, T<:Number} = a.B .= a.B * (1/marginalize(a, collect(1:d)))
normalize!(a::PSDModelPolynomial{d, T}, measure::Function) where {d, T<:Number} = a.B .= a.B * (1/marginalize(a, collect(1:d), measure))