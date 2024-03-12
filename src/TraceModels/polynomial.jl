

abstract type OrthonormalTraceModel{T, S} <: TraceModel{T} end

function ΦΦT(a::OT, x::PSDdata{T}) where {T<:Number, OT<:OrthonormalTraceModel{T, Nothing}}
    return a.ΦΦT(x)
end

function ΦΦT(a::OT, x::PSDdata{T}) where {T<:Number, OT<:OrthonormalTraceModel{<:Number, Nothing}}
    return a.ΦΦT(x)
end

function ΦΦT(a::OrthonormalTraceModel{T, <:OMF}, x::PSDdata{T}) where {T<:Number}
    return a.ΦΦT(ξ(a.mapping, x)) * (1/x_deriv_prod(a.mapping, ξ(a.mapping, x), a.ΦΦT.int_dim))
end

struct PolynomialTraceModel{T<:Number, S<:Union{Nothing, OMF{T}}} <: OrthonormalTraceModel{T, S}
    B::Hermitian{T, <:AbstractMatrix{T}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    ΦΦT::SquaredPolynomialMatrix{<:Any, T}
    mapping::S
    function PolynomialTraceModel(B::Hermitian{T, <:AbstractMatrix{T}},
                                ΦΦT::SquaredPolynomialMatrix{<:Any, T}
                    ) where {T<:Number}
        new{T, Nothing}(B, ΦΦT, nothing)
    end
    function PolynomialTraceModel(B::Hermitian{T, <:AbstractMatrix{T}},
                                ΦΦT::SquaredPolynomialMatrix{<:Any, T},
                                mapping::S
                    ) where {T<:Number, S}
        new{T, S}(B, ΦΦT, mapping)
    end
end
