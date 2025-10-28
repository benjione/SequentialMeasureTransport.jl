

struct TemperedLimitStateFunctionBridgingDensity{d, T} <: BridgingDensity{d, T}
    density         # the density function, can be normalized (a priori) or unnormalized (a posteriori)
    g               # limit state function
    h               # function to smooth original h
    β_list          # list of params
end

function TemperedLimitStateFunctionBridgingDensity{d, T}(density, g, L::Int; a=0.8, b=1.0) where {d, T}
    h = (x, β)->0.5*(1+tanh(-x/β))
    β_list = [b*a^(i-1) for i=1:L]
    TemperedLimitStateFunctionBridgingDensity{d, T}(density, g, h, β_list)
end

function evaluate_bridge(bridge::TemperedLimitStateFunctionBridgingDensity{<:Any, T}, x::PSDdata{T}, i::Int) where {T<:Number}
    β = bridge.β_list[i]
    g_x = bridge.g(x)
    bridge.h(g_x, β) * bridge.density(x)
end