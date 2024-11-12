

struct DiffusionBrigdingDensity{d, T} <: BridgingDensity{d, T}
    target_density::Function
    t_vec::Vector{T}             # time vector of diffusion steps
    σ::Vector{T}                 # diffusion coefficient
    μ::Vector{T}                 # mean of the diffusion
    function DiffusionBrigdingDensity{d}(target_density::Function, 
                        t_vec::Vector{T},
                        σ::T) where {d, T<:Number}
        new{d, T}(target_density, t_vec, σ*ones(T, d), zeros(T, d))
    end
    function DiffusionBrigdingDensity{d}(target_density::Function, 
        t_vec::Vector{T},
        σ::Vector{T}, μ::Vector{T}) where {d, T<:Number}
        new{d, T}(target_density, t_vec, σ, μ)
    end
    function DiffusionBrigdingDensity{d}(target_density::Function, 
        t_vec::Vector{T},
        σ::T, μ::Vector{T}) where {d, T<:Number}
        new{d, T}(target_density, t_vec, σ*ones(T, d), μ)
    end
    function DiffusionBrigdingDensity{d}(target_density::Function, 
                        β::T,
                        N::Int) where {d, T<:Number}
        t_vec = choosing_timesteps(β, d, N)
        new{d, T}(target_density, t_vec, ones(T, d), zeros(T, d))
    end
    function DiffusionBrigdingDensity{d, T}() where {d, T<:Number}
        new{d, T}(x->1.0, T[], ones(T, d), zeros(T, d))
    end
    function DiffusionBrigdingDensity{d, T}(σ::T, μ::Vector{T}) where {d, T<:Number}
        @assert length(μ) == d
        new{d, T}(x->1.0, T[], σ*ones(T, d), μ)
    end
    function DiffusionBrigdingDensity{d, T}(σ::Vector{T}, μ::Vector{T}) where {d, T<:Number}
        @assert length(σ) == d
        @assert length(μ) == d
        new{d, T}(x->1.0, T[], σ, μ)
    end
end

function add_timestep!(bridge::DiffusionBrigdingDensity{d, T}, β::T) where {d, T<:Number}
    
    next_t(t_previous) = begin
        return -0.5 * log(1.0 -
                (1/β^(2/d)) * (1.0 - exp(-2.0 * t_previous)))
    end

    t_ℓ = if length(bridge.t_vec) == 0
        -0.5 * log(1.0 - (1/β^(2/d)))
    else
        next_t(bridge.t_vec[end])
    end

    push!(bridge.t_vec, t_ℓ)
    return t_ℓ
end

function choosing_timesteps(β::T, d, N::Int) where { T<:Number}
    @assert β > 1.0
    ## by Proposition 6
    next_t(t_previous) = begin
        return -0.5 * log(1.0 -
                (1/β^(2/d)) * (1.0 - exp(-2.0 * t_previous)))
    end
    next_t() = return -0.5 * log(1.0 - (1/β^(2/d)))
    t_vec = Vector{T}(undef, N)
    t_vec[1] = next_t()
    for i=2:(N-1)
        t_vec[i] = next_t(t_vec[i-1])
    end
    t_vec[end] = 0.0
    return t_vec
end

function evolve_samples(bridge::DiffusionBrigdingDensity{<:Any, T},
                        X::PSDDataVector{T}, 
                        t::T; enrichment::Int=1) where {T<:Number}
    X_t = Vector{Vector{T}}(undef, enrichment*length(X))

    for i=1:length(X)
        for j=1:enrichment
            X_t[(i-1)*enrichment + j] = exp(-t) * X[i] .+ sqrt(1 - exp(-t)) * bridge.σ .* randn(T, size(X[i]))
        end
    end
    return X_t
end