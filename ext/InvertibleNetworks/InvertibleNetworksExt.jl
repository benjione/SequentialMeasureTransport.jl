module InvertibleNewtorksExt

using InvertibleNetworks

import SequentialMeasureTransport as SMT

struct InvertibleNetworksMapping{d, dC, T, NT} <: SMT.ConditionalMapping{d, dC, T}
    network::NT
    forward::Function
    inverse::Function
    function InvertibleNetworksMapping{d, dC, T}(network::NT,
                        forward::Function,
                        inverse::Function) where {NT, d, dC, T<:Number}
        new{d, dC, T, NT}(network, forward, inverse)
    end
end

SMT.Sampler(network, forward, inverse, d, T; dC=0) = InvertibleNetworksMapping{d, dC, T}(network, forward, inverse)

function SMT.pushforward(m::InvertibleNetworksMapping{<:Any, <:Any, T}, x::SMT.PSDdata{T}) where {T <: Number}
    y = m.forward(reshape(x, 1, 1, length(x), 1))[1]
    return reshape(y, length(y))
end

function SMT.pullback(m::InvertibleNetworksMapping{<:Any, <:Any, T}, y::SMT.PSDdata{T}) where {T <: Number}
    x = m.inverse(reshape(y, 1, 1, length(y), 1))
    return reshape(x, length(x))
end

function SMT.log_Jacobian(m::InvertibleNetworksMapping{<:Any, <:Any, T}, x::SMT.PSDdata{T}) where {T <: Number}
    return m.forward(reshape(x, 1, 1, length(x), 1))[2]
end

function SMT.inverse_log_Jacobian(m::InvertibleNetworksMapping{<:Any, <:Any, T}, x::SMT.PSDdata{T}) where {T <: Number}
    return 1/SMT.log_Jacobian(m, SMT.pushforward(m, x))
end

function SMT.Jacobian(m::InvertibleNetworksMapping{<:Any, <:Any, T}, x::SMT.PSDdata{T}) where {T <: Number}
    return exp(SMT.log_Jacobian(m, x))
end

function SMT.inverse_Jacobian(m::InvertibleNetworksMapping{<:Any, <:Any, T}, x::SMT.PSDdata{T}) where {T <: Number}
    return exp(SMT.inverse_log_Jacobian(m, x))
end

end