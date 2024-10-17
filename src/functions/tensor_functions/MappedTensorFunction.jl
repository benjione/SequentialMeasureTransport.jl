

struct MappedTensorFunction{d , T , M<:ConditionalMapping{d, <:Any, T}, S<:Tensorizer{d}} <: TensorFunction{d, T, S}
    tf::TensorFunction{d, T, S}
    mapping::M
end

function (p::MappedTensorFunction{<:Any, T, M})(x::PSDdata{T}) where {T<:Number, M}
    return p.tf(pushforward(p.mapping, x)) * Jacobian(p.mapping, x)
end

function (p::MappedTensorFunction{<:Any, T1, M})(x::PSDdata{T2}) where {T1<:Number, T2<:Number, M}
    return p.tf(pushforward(p.mapping, x)) * Jacobian(p.mapping, x)
end

