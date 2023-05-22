

struct logarithmicOMF{T} <: OMF{T} end

function x(::logarithmicOMF{T} ,ξ::T) where {T<:Number}
    tanh(ξ)
end

function ξ(::logarithmicOMF{T}, x::T) where {T<:Number}
    0.5*log((1+x)/(1-x))
end

function x_deriv(::logarithmicOMF{T}, ξ::T) where {T<:Number}
    1-tanh(ξ)^2
end

function ξ_deriv(::logarithmicOMF{T}, x::T) where {T<:Number}
    1/(1-x^2)
end