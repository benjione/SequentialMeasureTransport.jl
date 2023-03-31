


struct TensorizedPolynomialTraceModel{T<:Number} <: TraceModel{T}
    B::Hermitian{Float64, Matrix{Float64}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    ΦΦT::SquaredPolynomialMatrix{<:Any, T}
    function TensorizedPolynomialTraceModel(B::Hermitian{Float64, Matrix{Float64}},
                                ΦΦT::SquaredPolynomialMatrix{<:Any, T}
                    ) where {T<:Number}
        new{T}(B, ΦΦT)
    end
end


function ΦΦT(a::TensorizedPolynomialTraceModel{T}, x::PSDdata{T}) where {T<:Number}
    return a.ΦΦT(x)
end