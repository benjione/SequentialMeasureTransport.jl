
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

function pushforward(
        m::ScalingReference{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    return (x .- m.L) ./ (m.R .- m.L)
end


function pullback(
        m::ScalingReference{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    return u .* (m.R .- m.L) .+ m.L
end


function Jacobian(
        mapping::ScalingReference{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    return 1/mapping.V
end


function inverse_Jacobian(
        mapping::ScalingReference{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    return mapping.V
end