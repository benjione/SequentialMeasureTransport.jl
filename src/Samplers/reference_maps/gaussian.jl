

"""
Map to a gaussian reference distribution.
"""
struct GaussianReference{d, T} <: ReferenceMap{d, T} end


function sample_reference(map::GaussianReference{d, T}) where {d, T<:Number}
    randn(T, d)
end

function sample_reference(map::GaussianReference{d, T}, n::Int) where {d, T<:Number}
    randn(T, d, n)
end

function pushforward(
        m::GaussianReference{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    return 0.5 * (1 .+ erf.(x ./ sqrt(2)))
end


function pullback(
        m::GaussianReference{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    return sqrt(2) * erfcinv.(2.0 .- 2*u)
end


function Jacobian(
        mapping::GaussianReference{d, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    return mapreduce(xi->Distributions.pdf(Distributions.Normal(0, 1), xi), *, x)
end


function inverse_Jacobian(
        mapping::GaussianReference{d, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    # inverse function theorem
    return 1/Jacobian(mapping, pullback(mapping, u))
end