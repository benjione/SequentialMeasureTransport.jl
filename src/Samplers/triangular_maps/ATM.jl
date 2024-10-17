

# abstract type ATM{d, dC, T} <: AbstractTriangularMap{d, dC, T} end

struct PolynomialATM{d, dC, T<:Number} <: AbstractTriangularMap{d, dC, T}
    f::Vector{<:TensorFunction{<:Any, T}}
    coeff::Vector{<:Vector{T}}
    g::Function
    variable_ordering::Vector{Int}
    function PolynomialATM(f::Vector{<:TensorFunction{<:Any, T}}, g::Function, variable_ordering::Vector{Int}, dC::Int) where {T<:Number}
        d = length(f)
        coeff = Vector{Vector{T}}(undef, d)
        for k=1:d
            coeff[k] = randn(T, length(f[k](rand(k))))
        end
        new{d, dC, T}(f, coeff, g, variable_ordering)
    end
    function PolynomialATM(f::Vector{<:TensorFunction}, g::Function, variable_ordering::Vector{Int})
        PolynomialATM(f, g, variable_ordering, 0)
    end
end

int_x, int_w = gausslegendre(50)
int_x .= int_x * 0.5 .+ 0.5
int_w .= int_w * 0.5
@inline MonotoneMap(sampler::PolynomialATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = MonotoneMap(sampler, x, k, sampler.coeff[k])
function MonotoneMap(sampler::PolynomialATM{d, <:Any, T}, x::PSDdata{T}, k::Int, 
                    coeff::AbstractVector{T}) where {d, T<:Number}
    f_part(z) = begin
        sampler.f[k]([x[1:k-1]; z])
    end
    f_partial(z::T) = FD.derivative(f_part, z)
    int_f(z::T) = sampler.g(dot(coeff, f_partial(z)))
    
    _int_x = int_x * x[k]
    _int_w = int_w * x[k]
    res = dot(coeff, sampler.f[k]([x[1:k-1]; 0]))
    for i=1:length(int_x)
        @inbounds res += _int_w[i] * int_f(_int_x[i])
    end
    return res
end

function ∇MonotoneMap(sampler::PolynomialATM{d, <:Any, T}, x::PSDdata{T}, k::Int, 
                    coeff::AbstractVector{T}) where {d, T<:Number}
    grad = zeros(size(coeff))
    ∇MonotoneMap!(grad, sampler, x, k, coeff)
    return grad
end

function ∇MonotoneMap!(grad::AbstractVector{T}, sampler::PolynomialATM{d, <:Any, T}, 
                    x::PSDdata{T}, k::Int, coeff::AbstractVector{T};
                    weight::T=one(T)) where {d, T<:Number}
    f_part(z) = begin
        sampler.f[k]([x[1:k-1]; z])
    end
    f_partial(z::T) = FD.derivative(f_part, z)
    int_f2!(tmp::AbstractVector{T}, z::T) = begin
        tmp .= f_partial(z)
        tmp .*= FD.derivative(sampler.g, dot(coeff, tmp))
        return nothing
    end
    _int_x = int_x * x[k]
    _int_w = int_w * x[k]
    tmp = similar(coeff)
    grad .+= weight * sampler.f[k]([x[1:k-1]; 0])
    for i=1:length(int_x)
        int_f2!(tmp, @inbounds _int_x[i])
        @inbounds grad .+= weight * _int_w[i] * tmp
    end
    return nothing
end

@inline ∂k_MonotoneMap(sampler::PolynomialATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = ∂k_MonotoneMap(sampler, x, k, sampler.coeff[k])
function ∂k_MonotoneMap(sampler::PolynomialATM{d, <:Any, T}, x::PSDdata{T}, k::Int, 
                            coeff::AbstractVector{T}) where {d, T<:Number}
    f_part(z) = sampler.f[k]([x[1:k-1]; z])
    f_partial(z::T) = FD.derivative(f_part, z)
    int_f(z::T) = sampler.g(dot(coeff, f_partial(z)))
    return int_f(x[k])
end

function ∇∂k_MonotoneMap(sampler::PolynomialATM{d, <:Any, T}, x::PSDdata{T}, k::Int, 
                        coeff::AbstractVector{T}) where {d, T<:Number}
    grad = zeros(size(coeff))
    ∇∂k_MonotoneMap!(grad, sampler, x, k, coeff)
    return grad
end

function ∇∂k_MonotoneMap!(grad::AbstractVector{T}, sampler::PolynomialATM{d, <:Any, T}, x::PSDdata{T}, k::Int, 
                coeff::AbstractVector{T}; weight::T=one(T)) where {d, T<:Number}
    f_part(z) = sampler.f[k]([x[1:k-1]; z])
    f_partial(z::T) = FD.derivative(f_part, z)
    tmp = f_partial(x[k])
    g_diff = FD.derivative(sampler.g, dot(coeff, tmp))
    grad .+= weight * g_diff * tmp
    return nothing
end

"""
Map of type
f1(x_{1:k-1}) + g(f2(x_{1:k-1})) * x_k
"""
struct PolynomialCouplingATM{d, dC, T} <: AbstractTriangularMap{d, dC, T}
    f1::AbstractVector{<:TensorFunction{<:Any, T}}
    f2::AbstractVector{<:TensorFunction{<:Any, T}}
    coeff::AbstractVector{<:AbstractMatrix{T}}
    g::Function
    # poly_measure::Function
    variable_ordering::Vector{Int}
end

function PolynomialCouplingATM(f1, f2, g, variable_ordering)
    d = length(f1)
    coeff = Vector{Matrix{Float64}}(undef, d)
    coeff = [k==1 ? randn(2, 1) : randn(Float64, 2, length(f1[k](rand(k-1)))) for k=1:d]
    # for k=1:d
    #     coeff[k] = randn(Float64, 2, length(f1[k](rand(k))))
    # end
    PolynomialCouplingATM{d, 0, Float64}(f1, f2, coeff, g, variable_ordering)
end

@inline MonotoneMap(sampler::PolynomialCouplingATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = MonotoneMap(sampler, x, k, sampler.coeff[k])
function MonotoneMap(sampler::PolynomialCouplingATM{d, <:Any, T}, x::PSDdata{T}, 
                    k::Int, coeff::AbstractMatrix{T2}) where {d, T<:Number, T2<:Number}
    if k==1
        return coeff[1, 1] + x[1]
    end
    # print("here ",  exp(-norm(x[1:k-1])^2/2),  exp(-norm(x[1:k-1])^2/2) * dot(coeff[2, :], sampler.f2[k](x[1:k-1])))
    return dot(coeff[1, :], sampler.f1[k](x[1:k-1])) + 
            sampler.g(dot(coeff[2, :], sampler.f2[k](x[1:k-1]))) * x[k]
end

function ∇MonotoneMap(sampler::PolynomialCouplingATM{d, <:Any, T}, x::PSDdata{T}, k::Int, 
                            coeff::AbstractMatrix{T2}) where {d, T<:Number, T2<:Number}
    if k==1
        return hcat([1.0], [0.0])'
    end
    g_diff = FD.derivative(sampler.g, dot(coeff[2, :], sampler.f2[k](x[1:k-1])))
    return hcat(sampler.f1[k](x[1:k-1]), g_diff * sampler.f2[k](x[1:k-1]) * x[k])'
end

@inline ∂k_MonotoneMap(sampler::PolynomialCouplingATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = ∂k_MonotoneMap(sampler, x, k, sampler.coeff[k])
function ∂k_MonotoneMap(sampler::PolynomialCouplingATM{d, <:Any, T}, x::PSDdata{T}, k::Int, coeff) where {d, T<:Number}
    if k==1
        return 1.0
    end
    return sampler.g(dot(coeff[2, :], sampler.f2[k](x[1:k-1])))
end

function ∇∂k_MonotoneMap(sampler::PolynomialCouplingATM{d, <:Any, T}, x::PSDdata{T}, k::Int, 
                        coeff::AbstractMatrix{T2}) where {d, T<:Number, T2<:Number}
    if k==1
        return hcat([0.0], [0.0])'
    end
    g_diff = FD.derivative(sampler.g, dot(coeff[2, :], sampler.f2[k](x[1:k-1])))
    return hcat(zeros(size(coeff, 2)), g_diff * sampler.f2[k](x[1:k-1]))'
end

"""
Defined on [0, 1]^d
of type
\\int_{0}^{x_k} Φ(x_{1:k-1}, z)' A Φ(x_{1:k-1}, z) dz with A ⪰ 0, tr(A) = 1

"""
struct SoSATM{d, dC, T} <: AbstractTriangularMap{d, dC, T}
    f::Vector{<:PSDModel{T}}
    f_int::Vector{<:TraceModel{T}}
    A_vec::Vector{<:Hermitian{T}}
    variable_ordering::Vector{Int}
    function SoSATM(f::Vector{<:PSDModel{T}}, variable_ordering::Vector{Int}) where {T}
        d = length(f)
        f_int = [integral(f[k], k) for k=1:length(f)]
        A_vec = [f[k].B for k=1:length(f)]
        new{d, 0, T}(f, f_int, A_vec, variable_ordering)
    end
end

@inline MonotoneMap(sampler::SoSATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = MonotoneMap(sampler, x, k, sampler.A_vec[k])
function MonotoneMap(sampler::SoSATM{d, <:Any, T}, x::PSDdata{T}, 
                    k::Int, coeff::AbstractMatrix{T2}) where {d, T<:Number, T2<:Number}
    return sampler.f_int[k](x[1:k], coeff)
end

function ∇MonotoneMap(sampler::SoSATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number}
    return parameter_gradient(sampler.f_int[k], x[1:k])
end

@inline ∂k_MonotoneMap(sampler::SoSATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = ∂k_MonotoneMap(sampler, x, k, sampler.A_vec[k])
function ∂k_MonotoneMap(sampler::SoSATM{d, <:Any, T}, x::PSDdata{T}, k::Int, coeff) where {d, T<:Number}
    return sampler.f[k](x[1:k], coeff)
end

function ∇∂k_MonotoneMap(sampler::SoSATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number}
    return parameter_gradient(sampler.f[k], x[1:k])
end