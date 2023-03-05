using ApproxFun

"""
A PSD model where the feature map is made out of polynomial functions such as
Chebyshev polynomials. There is an own implementation in order to provide
optimal weighted sampling, closed form derivatives and integration, etc.
As a polynomial package, ApproxFun is used.
"""
struct PSDModelFMPolynomial{T<:Number} <: AbstractPSDModelFM{T}
    B::Hermitian{Float64, Matrix{Float64}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    Π::Space                                # Polynomial space of the feature map
    Φ::Function                             # Φ(x) is the feature map
    function PSDModelFMPolynomial{T}(B::Hermitian{Float64, Matrix{Float64}}, 
                                    Π::Space
                    ) where {T<:Number}
        N = size(B, 1)
        map = [Fun(Π, Float64[zeros(d), 1.0]) for d=0:(N-1)]
        Φ(x) = map(f-> f(x), f in map)
        new{T}(B, Π, Φ)
    end
end
