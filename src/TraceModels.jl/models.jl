
include("polynomial.jl")
include("polynomial_tensorized.jl")

## trace model evaluation
function (a::TraceModel{T})(x::PSDdata{T}) where {T<:Number}
    M = ΦΦT(a, x)
    return tr(a.B * M)
end

function (a::PolynomialTraceModel)(x::PSDdata{T}, B::AbstractMatrix{T}) where {T<:Number}
    M = ΦΦT(a, x)
    return tr(B * M)
end