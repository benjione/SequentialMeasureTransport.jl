

struct AlgebraicBridgingDensity{d, T} <: BridgingDensity{d, T}
    target_density::Function
    reference_density::Function
    β_list::Vector{T}
    function AlgebraicBridgingDensity{d}(target_density::Function, 
                                        reference_density::Function,
                                         β_list::Vector{T}) where {d, T<:Number}
        new{d, T}(target_density, reference_density, β_list)
    end
    function AlgebraicBridgingDensity{d}(target_density::Function, 
                                         β_list::Vector{T}) where {d, T<:Number}
        AlgebraicBridgingDensity{d}(target_density, x->one(T), β_list)
    end
end

function evaluate_bridge(bridge::AlgebraicBridgingDensity{d, T}, 
    x::PSDdata{T}, 
    k::Int) where {d, T<:Number}
    return bridge.target_density(x)^(bridge.β_list[k]) * bridge.reference_density(x)^(1 - bridge.β_list[k])
end

function evaluate_bridge(bridge::AlgebraicBridgingDensity{d, T}, 
    X::PSDDataVector{T}, 
    k::Int) where {d, T<:Number}
    return bridge.target_density(X).^(bridge.β_list[k]) .* bridge.reference_density(X).^(1 - bridge.β_list[k])
end