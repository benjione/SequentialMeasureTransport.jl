


struct PSDModelFM{T<:Number} <: PSDModel{T}
    B::Hermitian{Float64, Matrix{Float64}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    ϕ                                       # ϕ(x) is the feature map
    function PSDModelFM(B::Hermitian{Float64, Matrix{Float64}}, 
                      ϕ
                    ) where {T<:Number}
        new{T}(B, ϕ)
    end
end


function (a::PSDModelFM)(x::PSDdata{T}) where {T<:Number}
    v = a.ϕ(x)
    return v' * a.B * v
end


function (a::PSDModelFM)(x::PSDdata{T}, B::AbstractMatrix{T}) where {T<:Number}
    v = a.ϕ(x)
    return v' * B * v
end
