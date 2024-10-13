
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

function Base.getindex(m::AlgebraicReference{d, dC, T}, I) where {d, dC, T}
    vars = collect(I)
    if length(vars) == 0
        return nothing
    end
    d_new = length(vars)
    dC_new = length(intersect(d-dC+1:d, vars))
    # @assert setdiff(1:d-dC, vars) == 1:dC_new
    return AlgebraicReference{d_new, dC_new, T}()
end

function _pushforward(m::AlgebraicReference{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number}
    return ((x./sqrt.(1 .+ x.^2)).+1.0)/2.0
end

function _pullback(m::AlgebraicReference{<:Any, <:Any, T}, u::PSDdata{T}) where {T<:Number}
    ξ = 2.0*(u .- 0.5)
    return ξ./sqrt.(1.0 .- ξ.^2)
end

function _Jacobian(m::AlgebraicReference{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number}
    return mapreduce(xi->0.5/(1+(xi)^2)^(3/2), *, x)
end

function _inverse_Jacobian(m::AlgebraicReference{<:Any, <:Any, T}, u::PSDdata{T}) where {T<:Number}
    return mapreduce(ui->2.0/(1-((2*ui - 1.0)^2))^(3/2), *, u)
end

function _log_Jacobian(m::AlgebraicReference{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number}
    return mapreduce(xi->log(0.5) - (3/2) * log(1+xi^2), +, x)
end

function _inverse_log_Jacobian(m::AlgebraicReference{<:Any, <:Any, T}, u::PSDdata{T}) where {T<:Number}
    return mapreduce(ui->log(2.0) - (3/2) * log(1.0-((2*ui - 1.0)^2)), +, u)
end


function SMT.pushforward(
        m::AlgebraicReference{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(x) == d
    return _pushforward(m, x)
end


function SMT.pullback(
        m::AlgebraicReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(u) == d
    return _pullback(m, u)
end

function SMT.Jacobian(
        m::AlgebraicReference{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(x) == d
    return _Jacobian(m, x)
end


function SMT.inverse_Jacobian(
        m::AlgebraicReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(u) == d
    return _inverse_Jacobian(m, u)
end

function SMT.log_Jacobian(
        m::AlgebraicReference{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(x) == d
    return _log_Jacobian(m, x)
end

function SMT.inverse_log_Jacobian(
        mapping::AlgebraicReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(u) == d
    return _inverse_log_Jacobian(mapping, u)
end


function SMT.marginal_pushforward(
        m::AlgebraicReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    return _pushforward(m, x)
end

function SMT.marginal_pullback(
        m::AlgebraicReference{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(u) == d-dC
    ξ = 2.0*(u .- 0.5)
    return _pullback(m, u)
end

function SMT.marginal_Jacobian(
        m::AlgebraicReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    return _Jacobian(m, x)
end

function SMT.marginal_inverse_Jacobian(
        mapping::AlgebraicReference{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(u) == d-dC
    return _inverse_Jacobian(mapping, u)
end

function SMT.marginal_log_Jacobian(
        m::AlgebraicReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    return _log_Jacobian(m, x)
end

function SMT.marginal_inverse_log_Jacobian(
        mapping::AlgebraicReference{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(u) == d-dC
    return _inverse_log_Jacobian(mapping, u)
end