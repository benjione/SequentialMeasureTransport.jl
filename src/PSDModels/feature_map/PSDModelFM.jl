
abstract type AbstractPSDModelFM{T} <: PSDModel{T} end

include("PSDModelOrthonormal.jl")

# kwargs definition of PSDModelKernel
const _PSDModelFM_kwargs =
        Symbol[]

struct PSDModelFM{T<:Number} <: AbstractPSDModelFM{T}
    B::Hermitian{T, <:AbstractMatrix{T}}  # B is the PSD so that f(x) = ∑_ij Φ(x)_i * B_ij * Φ(x)_j
    Φ::Function                             # Φ(x) is the feature map
    function PSDModelFM{T}(B::Hermitian{T, <:AbstractMatrix{T}}, 
                    Φ
                    ) where {T<:Number}
        new{T}(B, Φ)
    end
end

@inline _of_same_PSD(a::PSDModelFM{T}, B::AbstractMatrix{T}) where {T<:Number} =
                                        PSDModelFM{T}(Hermitian(B), a.Φ)

"""
Φ(a::PSDModelFM, x::PSDdata{T}) where {T<:Number}

Returns the feature map of the PSD model at x.
"""
@inline function Φ(a::AbstractPSDModelFM{T}, x::PSDdata{T}) where {T<:Number}
    return a.Φ(x)
end

@inline function Φ(a::AbstractPSDModelFM{T}, x::PSDdata{T1}) where {T<:Number, T1}
    return a.Φ(x)
end

@inline function Φ(a::AbsPSD, x::PSDdata{T}) where {T<:Number, AbsPSD<:AbstractPSDModelFM{<:Number}}
    return a.Φ(x)
end
