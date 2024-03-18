
"""
Map to an algebraic reference distribution.
"""
struct AlgebraicReference{d, T} <: ReferenceMap{d, T} 
    function AlgebraicReference{d,T}() where {d, T<:Number}
        new{d, T}()
    end
end


function SequentialMeasureTransport.pushforward(
        m::AlgebraicReference{<:Any, T}, 
        x::PSDdata{T}
    ) where {T<:Number}
    return ((x./sqrt.(1 .+ x.^2)).+1.0)/2.0
end


function SequentialMeasureTransport.pullback(
        m::AlgebraicReference{<:Any, T}, 
        u::PSDdata{T}
    ) where {T<:Number}
    ξ = 2.0*(u .- 0.5)
    return ξ./sqrt.(1.0 .- ξ.^2)
end


function SequentialMeasureTransport.Jacobian(
        m::AlgebraicReference{<:Any, T}, 
        x::PSDdata{T}
    ) where {T<:Number}
    return mapreduce(xi->0.5/(1+(xi)^2)^(3/2), *, x)
end


function SequentialMeasureTransport.inverse_Jacobian(
        mapping::AlgebraicReference{<:Any, T}, 
        u::PSDdata{T}
    ) where {T<:Number}
    # inverse function theorem
    return mapreduce(ui->2.0/(1-((2*ui - 1.0)^2))^(3/2), *, u)
end
