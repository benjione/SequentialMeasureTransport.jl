"""
Linear Scaling independently in all directions.
"""
struct ScalingReference{d, T} <: ReferenceMap{d, T}
    L::Vector{T}
    R::Vector{T}
    V::T
    function ScalingReference{d}(L::Vector{T}, R::Vector{T}) where {d, T<:Number}
        @assert length(L) == d
        @assert length(R) == d
        @assert all(L .< R)
        V = prod(R .- L)
        new{d, T}(L, R, V)
    end
end

function ScalingReference(model::PSDModelOrthonormal{d, T, S}) where {d, T, S}
    L = domain_interval_left(model)
    R = domain_interval_right(model)
    ScalingReference{d}(L, R)
end



## Interface implementation

function PSDModels.pushforward(
        m::ScalingReference{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    d2 = length(x)
    return (x .- m.L[1:d2]) ./ (m.R[1:d2] .- m.L[1:d2])
end


function PSDModels.pullback(
        m::ScalingReference{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    d2 = length(u)
    return u .* (m.R[1:d2] .- m.L[1:d2]) .+ m.L[1:d2]
end


function Jacobian(
        mapping::ScalingReference{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    d2 = length(x)
    if d2 < d
        return 1/prod(mapping.R[1:d2] .- mapping.L[1:d2])
    end
    return 1/mapping.V
end


function inverse_Jacobian(
        mapping::ScalingReference{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    d2 = length(u)
    if d2 < d
        return prod(mapping.R[1:d2] .- mapping.L[1:d2])
    end
    return mapping.V
end