"""
Linear Scaling independently in all directions.
"""
struct ScalingReference{d, dC, T} <: ReferenceMap{d, dC, T}
    L::Vector{T}
    R::Vector{T}
    V::T
    V_margin::T
    function ScalingReference{d}(L::Vector{T}, R::Vector{T}) where {d, T<:Number}
        @assert length(L) == d
        @assert length(R) == d
        @assert all(L .< R)
        V = prod(R .- L)
        new{d, 0, T}(L, R, V, V)
    end
    function ScalingReference{d, dC}(L::Vector{T}, R::Vector{T}) where {d, dC, T<:Number}
        @assert length(L) == d
        @assert length(R) == d
        @assert all(L .< R)
        V = prod(R .- L)
        new{d, dC, T}(L, R, V, prod(R[1:d-dC] .- L[1:d-dC]))
    end
    function ScalingReference(L::Vector{T}, R::Vector{T}) where {T<:Number}
        ScalingReference{length(L)}(L, R)
    end
end

function ScalingReference(model::PSDModelOrthonormal{d, T, S}) where {d, T, S}
    L = domain_interval_left(model)
    R = domain_interval_right(model)
    ScalingReference{d}(L, R)
end



## Interface implementation

function SMT.pushforward(
        m::ScalingReference{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(x) == d
    return (x - m.L) ./ (m.R - m.L)
end


function SMT.pullback(
        m::ScalingReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(u) == d
    return u .* (m.R - m.L) + m.L
end


function SMT.Jacobian(
        mapping::ScalingReference{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(x) == d
    return 1/mapping.V
end


function SMT.inverse_Jacobian(
        mapping::ScalingReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(u) == d
    return mapping.V
end

function SMT.marginal_pushforward(
        m::ScalingReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    return (x .- m.L[1:dC]) ./ (m.R[1:dC] .- m.L[1:dC])
end

function SMT.marginal_pullback(
        m::ScalingReference{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(u) == d-dC
    return u .* (m.R[1:dC] .- m.L[1:dC]) .+ m.L[1:dC]
end

function SMT.marginal_Jacobian(
        mapping::ScalingReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    1/mapping.V_margin
end

function SMT.marginal_inverse_Jacobian(
        mapping::ScalingReference{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(u) == d-dC
    mapping.V_margin
end