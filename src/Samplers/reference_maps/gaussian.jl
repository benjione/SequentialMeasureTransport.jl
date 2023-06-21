

"""
Map to a gaussian reference distribution.
"""
struct GaussianReference{d, T} <: ReferenceMap{d, T} 
    σ::T
    function GaussianReference{d,T}(σ::T) where {d, T<:Number}
        new{d, T}(σ)
    end
    function GaussianReference{d,T}() where {d, T<:Number}
        new{d, T}(1.0)
    end
end


function sample_reference(map::GaussianReference{d, T}) where {d, T<:Number}
    randn(T, d) * map.σ
end

function sample_reference(map::GaussianReference{d, T}, n::Int) where {d, T<:Number}
    eachcol(randn(T, d, n) * map.σ)
end

function pushforward(
        m::GaussianReference{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    return 0.5 * (1 .+ erf.(x ./ (m.σ * sqrt(2))))
end


function pullback(
        m::GaussianReference{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    return sqrt(2) * m.σ * erfcinv.(2.0 .- 2*u)
end


function Jacobian(
        m::GaussianReference{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    return mapreduce(xi->Distributions.pdf(Distributions.Normal(0, m.σ), xi), *, x)
end


function inverse_Jacobian(
        mapping::GaussianReference{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    # inverse function theorem
    return 1/Jacobian(mapping, pullback(mapping, u))
end