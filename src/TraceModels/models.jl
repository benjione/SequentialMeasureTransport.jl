
include("polynomial.jl")

## trace model evaluation
function (a::TraceModel{T})(x::PSDdata{T}) where {T<:Number}
    M = ΦΦT(a, x)
    return tr(a.B * M)
end

function (a::TraceModel{T})(x::PSDdata{T}, B::AbstractMatrix{T}) where {T<:Number}
    M = ΦΦT(a, x)
    return tr(B * M)
end