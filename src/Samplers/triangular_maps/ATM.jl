

# abstract type ATM{d, dC, T} <: AbstractTriangularMap{d, dC, T} end

struct PolynomialATM{d, dC, T<:Number} <: AbstractTriangularMap{d, dC, T}
    f::Vector{<:FMTensorPolynomial{<:Any, T}}
    coeff::Vector{<:Vector{T}}
    g::Function
    variable_ordering::Vector{Int}
    function PolynomialATM(f::Vector{FMTensorPolynomial{<:Any, T}}, g::Function, variable_ordering::Vector{Int}, dC::Int) where {T<:Number}
        d = length(f)
        coeff = Vector{Vector{T}}(undef, d)
        for k=1:d
            coeff[k] = randn(T, length(f[k](rand(k))))
        end
        new{d, dC, T}(f, coeff, g, variable_ordering)
    end
    function PolynomialATM(f::Vector{FMTensorPolynomial}, g::Function, variable_ordering::Vector{Int})
        PolynomialATM(f, g, variable_ordering, 0)
    end
end

int_x, int_w = gausslegendre(50)
int_x .= int_x * 0.5 .+ 0.5
int_w .= int_w * 0.5
@inline MonotoneMap(sampler::PolynomialATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = MonotoneMap(sampler, x, k, sampler.coeff[k])
function MonotoneMap(sampler::PolynomialATM{d, <:Any, T}, x::PSDdata{T}, k::Int, coeff) where {d, T<:Number}
    f_part(z) = begin
        sampler.f[k]([x[1:k-1]; z])
    end
    f_partial(z) = FD.derivative(f_part, z)
    int_f(z) = sampler.g(dot(coeff, f_partial(z)))
    
    _int_x = copy(int_x)
    _int_x .= _int_x * x[k]
    _int_w = int_w * x[k]


    int_part = sum(_int_w .* int_f.(_int_x))
    return dot(coeff, sampler.f[k]([x[1:k-1]; 0])) + int_part
end

@inline ∂k_MonotoneMap(sampler::PolynomialATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = ∂k_MonotoneMap(sampler, x, k, sampler.coeff[k])
function ∂k_MonotoneMap(sampler::PolynomialATM{d, <:Any, T}, x::PSDdata{T}, k::Int, coeff) where {d, T<:Number}
    f_part(z) = sampler.f[k]([x[1:k-1]; z])
    f_partial(z) = FD.derivative(f_part, z)
    int_f(z) = sampler.g(dot(coeff, f_partial(z)))
    return int_f(x[k])
end

"""
Map of type
f1(x_{1:k-1}) + g(f2(x_{1:k-1})) * x_k
"""
struct PolynomialCouplingATM{d, dC, T} <: AbstractTriangularMap{d, dC, T}
    f1::AbstractVector{<:FMTensorPolynomial{<:Any, T}}
    f2::AbstractVector{<:FMTensorPolynomial{<:Any, T}}
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
            (sampler.g(dot(coeff[2, :], sampler.f2[k](x[1:k-1]))) + T(1e-5)) * x[k]
end

@inline ∂k_MonotoneMap(sampler::PolynomialCouplingATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = ∂k_MonotoneMap(sampler, x, k, sampler.coeff[k])
function ∂k_MonotoneMap(sampler::PolynomialCouplingATM{d, <:Any, T}, x::PSDdata{T}, k::Int, coeff) where {d, T<:Number}
    if k==1
        return 1.0
    end
    return (sampler.g(dot(coeff[2, :], sampler.f2[k](x[1:k-1]))) + T(1e-5))
end


"""
Defined on [0, 1]^d
of type
\\int_{0}^{x_k} Φ(x_{1:k-1}, z)' A Φ(x_{1:k-1}, z) dz with A ⪰ 0, tr(A) = 1

"""
struct SoSATM{d, dC, T} <: TriangularMap{d, dC, T}

end