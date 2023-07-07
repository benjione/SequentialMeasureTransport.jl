

struct AlgebraicBridgingDensity{d, T} <: BridgingDensity{d, T}
    target_density::Function
    β_list::Vector{T}
    function AlgebraicBridgingDensity{d}(target_density::Function, β_list::Vector{T}) where {d, T<:Number}
        new{d, T}(target_density, β_list)
    end
end

