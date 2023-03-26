

struct PolynomialTraceModel{T<:Number} <: TraceModel{T}
    B::Hermitian{Float64, Matrix{Float64}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    Π::Space                                # Polynomial space of the feature map
    func_mat::Matrix{Fun}                   # Φ(x) Φ(x)^T is the feature map in matrix form
    function PolynomialTraceModel{T}(B::Hermitian{Float64, Matrix{Float64}}, 
                                    Π::Space,
                                    func_mat::Matrix{<:Fun}
                    ) where {T<:Number}
        @assert size(func_mat, 1) == size(B, 1)
        new{T}(B, Π, func_mat)
    end
end

@inline _of_same_PSD(a::PolynomialTraceModel{T}, B::AbstractMatrix{T}) where {T<:Number} =
            PolynomialTraceModel{T}(Hermitian(B), a.Π, a.func_mat)

"""
function ΦΦT(a::PSDModelFMMatPolynomial{T}, x::PSDdata{T}) where {T<:Number}

Returns ΦΦT  where Φ is the feature map of the PSD model at x.
"""
function ΦΦT(a::PolynomialTraceModel{T}, x::PSDdata{T}) where {T<:Number}
    return map(f-> f(x), a.func_mat)
end

"""
integral(a::PSDModelFMPolynomial{T}, dim::AbstractVector)

Build an integral ``\\int_a^{x_n} p(x_1,...,z,...) dz`` of the PSD model over the given dimensions.
"""
function integral(a::PolynomialTraceModel{T}, dim::AbstractVector) where {T<:Number}
    IntOp = length(dim)>1 ? Integral(a.Π, dim) : Integral(a.Π)
    K = map(f-> IntOp * f, a.func_mat)

    return PolynomialTraceModel{T}(copy(a.B), a.Π, K)
end

function marginalize(a::PolynomialTraceModel{T}, dim::AbstractVector) where {T<:Number}
    IntOp = length(dim)>1 ? DefiniteIntegral(a.Π, dim) : DefiniteIntegral(a.Π)
    K = map(f-> IntOp * f, a.func_mat)
    Π = K[1,1].space
    return PolynomialTraceModel{T}(copy(a.B), Π, K)
end