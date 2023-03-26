


struct TensorizedPolynomialTraceModel{T<:Number}
    B::Hermitian{Float64, Matrix{Float64}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    M::Matrix{Function}
    function PSDModelFMMatTensorPolynomial{T}(B::Hermitian{Float64, Matrix{Float64}},
                                    M::Matrix{<:Function}
                    ) where {T<:Number}
        @assert size(M, 1) == size(B, 1)
        new{T}(B, M)
    end
end


function ΦΦT(a::TensorizedPolynomialTraceModel{T}, x::PSDdata{T}) where {T<:Number}
    return map(f->f(x), a.M)
end