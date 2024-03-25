

"""
Map to a gaussian reference distribution.
"""
struct GaussianReference{d, dC, T} <: ReferenceMap{d, dC, T} 
    σ::T
    function GaussianReference{d,T}(σ::T) where {d, T<:Number}
        new{d, 0, T}(σ)
    end
    function GaussianReference{d,T}() where {d, T<:Number}
        new{d, 0, T}(one(T))
    end
    function GaussianReference{d,dC,T}(σ::T) where {d, dC, T<:Number}
        new{d, dC, T}(σ)
    end
    function GaussianReference{d,dC,T}() where {d, dC, T<:Number}
        new{d, dC, T}(one(T))
    end
end

function Distributions.pdf(Rmap::GaussianReference{d, <:Any, T}, x::PSDdata{T}) where {d, T<:Number}
    @assert length(x) == d
    mapreduce(xi->Distributions.pdf(Distributions.Normal(0, Rmap.σ), xi), *, x)
end

function sample_reference(map::GaussianReference{d, <:Any, T}) where {d, T<:Number}
    randn(T, d) * map.σ
end

function sample_reference(map::GaussianReference{d, <:Any, T}, n::Int) where {d, T<:Number}
    eachcol(randn(T, d, n) * map.σ)
end

function SMT.pushforward(
        m::GaussianReference{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(x) == d
    return 0.5 * (1 .+ erf.(x ./ (m.σ * sqrt(2))))
end


function SMT.pullback(
        m::GaussianReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(u) == d
    return sqrt(2) * m.σ * erfcinv.(2.0 .- 2*u)
end


function SMT.Jacobian(
        m::GaussianReference{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(x) == d
    return mapreduce(xi->Distributions.pdf(Distributions.Normal(0, m.σ), xi), *, x)
end


function SMT.inverse_Jacobian(
        mapping::GaussianReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    # inverse function theorem
    return 1/Jacobian(mapping, pullback(mapping, u))
end

function SMT.marginal_pushforward(
        m::GaussianReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    return 0.5 * (1 .+ erf.(x ./ (m.σ * sqrt(2))))
end

function SMT.marginal_pullback(
        m::GaussianReference{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(u) == d-dC
    return sqrt(2) * m.σ * erfcinv.(2.0 .- 2*u)
end

function SMT.marginal_Jacobian(
        m::GaussianReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    mapreduce(xi->Distributions.pdf(Distributions.Normal(0, m.σ), xi), *, x)
end

function SMT.marginal_inverse_Jacobian(
        mapping::GaussianReference{<:Any, <:Any, T}, 
        u::PSDdata{T}
    ) where {T<:Number}
    # inverse function theorem
    return 1/SMT.marginal_Jacobian(mapping, SMT.marginal_pullback(mapping, u))
end