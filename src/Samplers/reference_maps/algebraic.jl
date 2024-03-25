
"""
Map to an algebraic reference distribution.
"""
struct AlgebraicReference{d, dC, T} <: ReferenceMap{d, dC, T} 
    function AlgebraicReference{d,T}() where {d, T<:Number}
        new{d, 0, T}()
    end
    function AlgebraicReference{d,dC,T}() where {d,dC, T<:Number}
        new{d, dC, T}()
    end
end


function SMT.pushforward(
        m::AlgebraicReference{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(x) == d
    return ((x./sqrt.(1 .+ x.^2)).+1.0)/2.0
end


function SMT.pullback(
        m::AlgebraicReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(u) == d
    ξ = 2.0*(u .- 0.5)
    return ξ./sqrt.(1.0 .- ξ.^2)
end


function SMT.Jacobian(
        m::AlgebraicReference{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(x) == d
    return mapreduce(xi->0.5/(1+(xi)^2)^(3/2), *, x)
end


function SMT.inverse_Jacobian(
        mapping::AlgebraicReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(u) == d
    return mapreduce(ui->2.0/(1-((2*ui - 1.0)^2))^(3/2), *, u)
end


function SMT.marginal_pushforward(
        m::AlgebraicReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    return ((x./sqrt.(1 .+ x.^2)).+1.0)/2.0
end

function SMT.marginal_pullback(
        m::AlgebraicReference{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(u) == d-dC
    ξ = 2.0*(u .- 0.5)
    return ξ./sqrt.(1.0 .- ξ.^2)
end

function SMT.marginal_Jacobian(
        m::AlgebraicReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    return mapreduce(xi->0.5/(1+(xi)^2)^(3/2), *, x)
end

function SMT.marginal_inverse_Jacobian(
        mapping::AlgebraicReference{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(u) == d-dC
    return mapreduce(ui->2.0/(1-((2*ui - 1.0)^2))^(3/2), *, u)
end