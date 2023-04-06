
include("polynomial.jl")

## trace model evaluation
function (a::TraceModel{T})(x::PSDdata{T}) where {T<:Number}
    M = ΦΦT(a, x)
    # return tr(a.B * M)
    # tr(B * M) = dot(B', M) = dot(B, M), but faster evaluation
    return dot(a.B, M)
end

function (a::TraceModel{T})(x::PSDdata{T}, B::AbstractMatrix{T}) where {T<:Number}
    M = ΦΦT(a, x)
    # return tr(B * M)
    # tr(B * M) = tr(M * B) = dot(M', B) = dot(M, B) , but faster evaluation
    return dot(M, B)
end