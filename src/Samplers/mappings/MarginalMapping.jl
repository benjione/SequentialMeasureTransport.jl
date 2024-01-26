

struct MarginalMapping{d,dC,T<:Number,
    dsub,dCsub,
    Mtype<:ConditionalMapping{dsub,dCsub,T}
} <: SubsetMapping{d,dC,T,dsub,dCsub}
    map::Mtype
    subvariables::AbstractVector{Int} # last dCsub variables are the conditional variables
    function MarginalMapping{d,dC}(mapping::ConditionalMapping{dsub,dCsub,T},
        subvariables::AbstractVector{Int}) where {d,dC,dsub,dCsub,T<:Number}
        @assert dsub < d
        @assert all(1 <= k <= d for k in subvariables)
        @assert dCsub ≤ dC
        @assert length(subvariables) == dsub
        new{d,dC,T,dsub,dCsub,typeof(mapping)}(mapping, subvariables)
    end
end

"""
Interface of Conditional Mapping
"""

function pushforward(m::MarginalMapping{d,dC,T,dsub,dCsub,Mtype},
        u::PSDdata{T}) where {d,dC,T<:Number,dsub,dCsub,Mtype<:ConditionalMapping{dsub,dCsub,T}}
    _u = copy(u)
    _u[m.subvariables] = pushforward(m.map, u[m.subvariables])
    return _u
end

function pullback(m::MarginalMapping{d,dC,T,dsub,dCsub,Mtype},
        u::PSDdata{T}) where {d,dC,T<:Number,dsub,dCsub,Mtype<:ConditionalMapping{dsub,dCsub,T}}
    _u = copy(u)
    _u[m.subvariables] = pullback(m.map, u[m.subvariables])
    return _u
end

function Jacobian(m::MarginalMapping{d,dC,T,dsub,dCsub,Mtype},
        x::PSDdata{T}) where {d,dC,T<:Number,dsub,dCsub,Mtype<:ConditionalMapping{dsub,dCsub,T}}
    return Jacobian(m.map, x[m.subvariables])
end

function inverse_Jacobian(m::MarginalMapping{d,dC,T,dsub,dCsub,Mtype},
        u::PSDdata{T}) where {d,dC,T<:Number,dsub,dCsub,Mtype<:ConditionalMapping{dsub,dCsub,T}}
    return inverse_Jacobian(m.map, u[m.subvariables])
end

function marg_pushforward(m::MarginalMapping{d,dC,T,dsub,dCsub,Mtype},
        u::PSDdata{T}) where {d,dC,T<:Number,dsub,dCsub,Mtype<:ConditionalMapping{dsub,dCsub,T}}
    u[m.subvariables[1:(dsub-dCsub)]] = marg_pushforward(m.map, u[m.subvariables[1:(dsub-dCsub)]])
    return u
end

function marg_pullback(m::MarginalMapping{d,dC,T,dsub,dCsub,Mtype},
        u::PSDdata{T}) where {d,dC,T<:Number,dsub,dCsub,Mtype<:ConditionalMapping{dsub,dCsub,T}}
    u[m.subvariables[1:(dsub-dCsub)]] = marg_pullback(m.map, u[m.subvariables[1:(dsub-dCsub)]])
    return u
end

function marg_Jacobian(m::MarginalMapping{d,dC,T,dsub,dCsub,Mtype},
        x::PSDdata{T}) where {d,dC,T<:Number,dsub,dCsub,Mtype<:ConditionalMapping{dsub,dCsub,T}}
    return marg_Jacobian(m.map, x[m.subvariables[1:(dsub-dCsub)]])
end

function marg_inverse_Jacobian(m::MarginalMapping{d,dC,T,dsub,dCsub,Mtype},
        u::PSDdata{T}) where {d,dC,T<:Number,dsub,dCsub,Mtype<:ConditionalMapping{dsub,dCsub,T}}
    return marg_inverse_Jacobian(m.map, u[m.subvariables[1:(dsub-dCsub)]])
end