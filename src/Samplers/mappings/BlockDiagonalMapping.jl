
"""
    A transport map, applying transport maps to subset of variables.
"""
struct BlockDiagonalMapping{d, dC, T<:Number, 
            M<:ConditionalMapping{<:Any,<:Any,T}
        } <: ConditionalMapping{d, dC, T}
    mappings::Vector{M}
    subvariables::Vector{Vector{Int}}
    function BlockDiagonalMapping{d, dC, T}(mappings::Vector{M},
            subvariables::Vector{Vector{Int}}
        ) where {d, dC, T<:Number, M<:ConditionalMapping{<:Any,<:Any,T}}
        @assert length(vcat(subvariables...)) == d
        @assert length(unique(vcat(subvariables...))) == d
        new{d, dC, T, M}(mappings, subvariables)
    end
end

function pushforward(m::BlockDiagonalMapping{<:Any, <:Any, T},
        u::PSDdata{T}) where {T<:Number}
    _u = similar(u)
    for (map, subvar) in zip(m.mappings, m.subvariables)
        _u[subvar] = pushforward(map, u[subvar])
    end
    return _u
end

function pullback(m::BlockDiagonalMapping{<:Any, <:Any, T},
        u::PSDdata{T}) where {T<:Number}
    _u = similar(u)
    for (map, subvar) in zip(m.mappings, m.subvariables)
        _u[subvar] = pullback(map, u[subvar])
    end
    return _u
end

function Jacobian(m::BlockDiagonalMapping{<:Any,<:Any,T},
        x::PSDdata{T}) where {T<:Number}
    ret = one(T)
    for (subvar, map) in zip(m.subvariables, m.mappings)
        ret *= Jacobian(map, x[subvar])
    end
    return ret
end

function inverse_Jacobian(m::BlockDiagonalMapping{<:Any,<:Any,T},
            u::PSDdata{T}) where {T<:Number}
    ret = one(T)
    for (subvar, map) in zip(m.subvariables, m.mappings)
        ret *= inverse_Jacobian(map, u[subvar])
    end
    return ret
end

function marginal_pushforward(m::BlockDiagonalMapping{d,dC,T},
        u::PSDdata{T}) where {d,dC,T<:Number}
    _u = similar(u)
    for (subvar, map) in zip(m.subvariables, m.mappings)
        subvar_cond = subvar[subvar.≤(d-dC)]
        u[subvar_cond] = marginal_pushforward(map, u[subvar_cond])
    end
    return _u
end

function marginal_pullback(m::BlockDiagonalMapping{d,dC,T},
        u::PSDdata{T}) where {d,dC,T<:Number}
    _u = similar(u)
    for (subvar, map) in zip(m.subvariables, m.mappings)
        subvar_cond = subvar[subvar.≤(d-dC)]
        u[subvar_cond] = marginal_pullback(map, u[subvar_cond])
    end
    return _u
end

function marginal_Jacobian(m::BlockDiagonalMapping{d,dC,T},
        x::PSDdata{T}) where {d,dC,T<:Number}
    ret = one(T)
    for (subvar, map) in zip(m.subvariables, m.mappings)
        subvar_cond = subvar[subvar.≤(d-dC)]
        ret *= marginal_Jacobian(map, x[subvar_cond])
    end
    return ret
end

function marginal_inverse_Jacobian(m::BlockDiagonalMapping{d,dC,T},
        u::PSDdata{T}) where {d,dC,T<:Number}
    ret = one(T)
    for (subvar, map) in zip(m.subvariables, m.mappings)
        subvar_cond = subvar[subvar.≤(d-dC)]
        ret *= marginal_inverse_Jacobian(map, u[subvar_cond])
    end
    return ret
end