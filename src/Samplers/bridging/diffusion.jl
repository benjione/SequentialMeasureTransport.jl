

struct DiffusionBrigdingDensity{d, T} <: BridgingDensity{d, T}
    target_density::Function
    t_vec::Vector{T}            # time vector of diffusion steps
    function DiffusionBrigdingDensity{d}(target_density::Function, t_vec::Vector{T}) where {d, T<:Number}
        new{d, T}(target_density, t_vec)
    end
end