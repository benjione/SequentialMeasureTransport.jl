
abstract type AbstractPSDModelFM{T} <: PSDModel{T} end

include("PSDModelFMPolynomial.jl")

# kwargs definition of PSDModelKernel
const _PSDModelFM_kwargs =
        Symbol[]

struct PSDModelFM{T<:Number} <: AbstractPSDModelFM{T}
    B::Hermitian{Float64, Matrix{Float64}}  # B is the PSD so that f(x) = ∑_ij Φ(x)_i * B_ij * Φ(x)_j
    Φ::Function                             # Φ(x) is the feature map
    function PSDModelFM{T}(B::Hermitian{Float64, Matrix{Float64}}, 
                    Φ
                    ) where {T<:Number}
        new{T}(B, Φ)
    end
end

"""
Φ(a::PSDModelFM, x::PSDdata{T}) where {T<:Number}

Returns the feature map of the PSD model at x.
"""
@inline function Φ(a::AbstractPSDModelFM, x::PSDdata{T}) where {T<:Number}
    return a.Φ(x)
end

