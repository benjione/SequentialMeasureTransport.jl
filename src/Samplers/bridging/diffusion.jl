

struct DiffusionBrigdingDensity{d, T} <: BridgingDensity{d, T}
    target_density::Function
    t_vec::Vector{T}            # time vector of diffusion steps
    σ::T                         # diffusion coefficient
    function DiffusionBrigdingDensity{d}(target_density::Function, 
                        t_vec::Vector{T},
                        σ::T) where {d, T<:Number}
        new{d, T}(target_density, t_vec, σ)
    end
end


function evolve_samples(bridge::DiffusionBrigdingDensity{<:Any, T},
                        X::PSDDataVector{T}, 
                        t::T) where {T<:Number}
    f(du, u, p, t) = (du .= -1.0 * u)
    g(du, u, p, t) = (du .= bridge.σ)
    tspan = (0.0, t)
    function evolve_X(x::PSDdata{T})
        prob = SDEProblem(f, g, x, tspan)
        sol = solve(prob)
        return sol.u[end]
    end
    X_t = Vector{Vector{T}}(undef, length(X))
    Threads.@threads for i=1:length(X)
        X_t[i] = evolve_X(X[i])
    end
    return X_t
end