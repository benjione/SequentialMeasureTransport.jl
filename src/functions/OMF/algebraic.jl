

struct algebraicOMF{T} <: OMF{T} end

function x(::algebraicOMF{T} ,ξ::T) where {T<:Number}
    ξ/sqrt(1-ξ^2)
end

function ξ(::algebraicOMF{T}, x::T) where {T<:Number}
    x/sqrt(1+x^2)
end

function x_deriv(::algebraicOMF{T}, ξ::T) where {T<:Number}
    1/(1-ξ^2)^(3/2)
end

function ξ_deriv(::algebraicOMF{T}, x::T) where {T<:Number}
    1/(1+x^2)^(3/2)
end