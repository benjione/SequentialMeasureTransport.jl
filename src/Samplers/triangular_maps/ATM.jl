

struct ATM{d, dC, T<:Number} <: AbstractTriangularMap{d, dC, T}
    f::Vector{<:FMTensorPolynomial{<:Any, T}}
    coeff::Vector{<:Vector{T}}
    g::Function
    variable_ordering::Vector{Int}
    function ATM(f::Vector{FMTensorPolynomial{<:Any, T}}, g::Function, variable_ordering::Vector{Int}, dC::Int) where {T<:Number}
        d = length(f)
        coeff = Vector{Vector{T}}(undef, d)
        for k=1:d
            coeff[k] = randn(T, length(f[k](rand(k))))
        end
        new{d, dC, T}(f, coeff, g, variable_ordering)
    end
    function ATM(f::Vector{FMTensorPolynomial}, g::Function, variable_ordering::Vector{Int})
        ATM(f, g, variable_ordering, 0)
    end
end


@inline MonotoneMap(sampler::ATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = map(sampler, x, k, sampler.coeff[k])
function MonotoneMap(sampler::ATM{d, <:Any, T}, x::PSDdata{T}, k::Int, coeff) where {d, T<:Number}
    f_part(z) = sampler.f[k]([x[1:k-1]; z])
    f_partial(z) = ForwarDiff.gradient(f_part, z)
    int_f(z) = sampler.g(dot(coeff, f_partial(z)))
    int_x, int_w = gausslegendre(20)
    int_part = sum(int_w .* int_f.(int_x))
    return dot(coeff, sampler.f[k]([x[1:k-1]; 0])) + int_part
end

@inline ∂k_MonotoneMap(sampler::ATM{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number} = jacobian(sampler, x, k, sampler.coeff[k])
function ∂k_MonotoneMap(sampler::ATM{d, <:Any, T}, x::PSDdata{T}, k::Int, coeff) where {d, T<:Number}
    f_part(z) = sampler.f[k]([x[1:k-1]; z])
    f_partial(z) = ForwarDiff.gradient(f_part, z)
    int_f(z) = sampler.g(dot(coeff, f_partial(z)))
    return int_f(x[k])
end


# function ML_fit!(sampler::ATM{d, <:Any, T}, X::PSDDataVector{T}) where {d, T<:Number}
#     for k=1:d
#         coeff_0 = sampler.coeff[k]
#         min_func(coeff::Vector{T}) = begin
#             (1/length(x)) * mapreduce(x->(0.5*MonotoneMap(sampler, x, k, coeff))^2 - log(∂k_MonotoneMap(sampler, x, k, coeff)), +, X)
#         end
#         sampler.coeff[k] = optimize(min_func, coeff_0, BFGS())
#     end
# end