


struct RareEventBridgingDensity{d, T} <: BridgingDensity{d, T}
    density         # the density function, can be normalized (a priori) or unnormalized (a posteriori)
    d_A             # distance of a point x to the boundary of A, negative if inside the set
    γ_list          # parameters γ for determining the smoothness
end


function evaluate_bridge(bridge::RareEventBridgingDensity{<:Any, T}, x::PSDdata{T}, i::Int) where {T<:Number}
    γ = bridge.γ_list[i]
    bridge.density(x) / (1 + exp(γ * bridge.d_A(x)))
end