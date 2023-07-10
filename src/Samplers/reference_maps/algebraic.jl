
"""
Map to an algebraic reference distribution.
"""
struct AlgebraicReference{d, T} <: ReferenceMap{d, T} 
    function AlgebraicReference{d,T}() where {d, T<:Number}
        new{d, T}()
    end
end


function PSDModels.pushforward(
        m::AlgebraicReference{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    return ((x./sqrt(1 .+ x.^2)).+1.0)/2.0
end


function PSDModels.pullback(
        m::AlgebraicReference{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    ξ = 2*(u .- 0.5)
    return ξ./sqrt.(1 .- ξ.^2)
end


function Jacobian(
        m::AlgebraicReference{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    return mapreduce(xi->2.0/(1+xi^2)^(3/2), *, x)
end


function inverse_Jacobian(
        mapping::AlgebraicReference{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    # inverse function theorem
    return mapreduce(ui->2.0/(1-ui^2)^(3/2), *, u)
end
