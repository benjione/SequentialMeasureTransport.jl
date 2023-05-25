

abstract type OMF{T} end

include("algebraic.jl")
include("logarithmic.jl")

# domain of x is R
# domain of ξ is [-1,1]

## intefacce for OMFs
function x(mapping::OMF{T} ,ξ::T) where {T<:Number}
    throw(error("Not implemented"))
end

function ξ(mapping::OMF{T}, x::T) where {T<:Number}
    throw(error("Not implemented"))
end

# derivative of x w.r.t. ξ
function x_deriv(mapping::OMF{T}, ξ::T) where {T<:Number}
    throw(error("Not implemented"))
end

# derivative of ξ w.r.t. x
function ξ_deriv(mapping::OMF{T}, x::T) where {T<:Number}
    throw(error("Not implemented"))
end


## common functions
x(mapping::OMF{T} ,ξ::PSDdata{T}) where {T<:Number} = x.(Ref(mapping), ξ)
ξ(mapping::OMF{T} ,x::PSDdata{T}) where {T<:Number} = ξ.(Ref(mapping), x)

# derivative of x w.r.t. ξ
function x_deriv_prod(mapping::OMF{T}, ξ::PSDdata{T}) where {T<:Number}
    mapreduce(x -> x_deriv(mapping, x), *, ξ)
end

# derivative of x w.r.t. ξ
function x_deriv_prod(mapping::OMF{T}, ξ::AbstractVector{T}, ignore_dim::Vector{Int}) where {T<:Number}
    mapreduce(x -> x_deriv(mapping, x), *, ξ[setdiff(1:end, ignore_dim)])
end

function x_deriv_prod(mapping::OMF{T}, ξ::T, ignore_dim::Vector{Int}) where {T<:Number}
    if length(ignore_dim) == 0
        return x_deriv(mapping, ξ)
    else
        return 1.0
    end
end

# derivative of ξ w.r.t. x
function ξ_deriv_prod(mapping::OMF{T}, X::PSDdata{T}) where {T<:Number}
    mapreduce(x -> ξ_deriv(mapping, x), *, X)
end