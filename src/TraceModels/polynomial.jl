


struct PolynomialTraceModel{T<:Number} <: TraceModel{T}
    B::Hermitian{T, <:AbstractMatrix{T}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    ΦΦT::SquaredPolynomialMatrix{<:Any, T}
    function PolynomialTraceModel(B::Hermitian{T, <:AbstractMatrix{T}},
                                ΦΦT::SquaredPolynomialMatrix{<:Any, T}
                    ) where {T<:Number}
        new{T}(B, ΦΦT)
    end
end


function ΦΦT(a::PolynomialTraceModel{T}, x::PSDdata{T}) where {T<:Number}
    return a.ΦΦT(x)
end