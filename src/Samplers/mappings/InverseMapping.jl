

"""
    InverseMapping{d, dC, T, F}

A reverse mapping is a mapping that is the inverse of another mapping.
"""
struct InverseMapping{d, dC, T, F} <: ConditionalMapping{d, dC, T}
    f::F
end

function InverseMapping(f::ConditionalMapping{d, dC, T}) where {d, dC, T} 
    InverseMapping{d, dC, T, typeof(f)}(f)
end

@inline pushforward(m::InverseMapping{<:Any, <:Any, T, F}, 
            u::PSDdata) where {T, F} = pullback(m.f, u)

@inline pullback(m::InverseMapping{<:Any, <:Any, T, F},
            x::PSDdata) where {T, F} = pushforward(m.f, x)

@inline Jacobian(m::InverseMapping{<:Any, <:Any, T, F},
            x::PSDdata) where {T, F} = inverse_Jacobian(m.f, x)

@inline inverse_Jacobian(m::InverseMapping{<:Any, <:Any, T, F},
            u::PSDdata) where {T, F} = Jacobian(m.f, u)