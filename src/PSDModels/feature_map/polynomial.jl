import Symbolics as Sym

"""
A PSD model where the feature map is made out of orthonormal 
polynomial functions such as Chebyshev polynomials. There is an own 
implementation in order to provide orthonormal polynomials,
optimal weighted sampling, closed form derivatives and integration, etc.
For orthogonal polynomials, the package ApproxFun is used.
"""
struct PSDModelPolynomial{d, T<:Number, S<:Union{Nothing, OMF{T}, ConditionalMapping{d, <:Any, T}}} <: PSDModelOrthonormal{d, T, S}
    B::Hermitian{T, <:AbstractMatrix{T}}
    Φ::FMTensorPolynomial{d, T}
    mapping::S
    function PSDModelPolynomial(B::Hermitian{T, <:AbstractMatrix{T}},
                                    Φ::FMTensorPolynomial{d, T}
                    ) where {d, T<:Number}
        
        new{d, T, Nothing}(B, Φ, nothing)
    end
    function PSDModelPolynomial{T}(B::Hermitian{T, <:AbstractMatrix{T}},
                                    Φ::FMTensorPolynomial{d, T}
                        )where {d, T<:Number}
        new{d, T, Nothing}(B, Φ, nothing)
    end
    function PSDModelPolynomial(B::Hermitian{T, <:AbstractMatrix{T}},
                                    Φ::FMTensorPolynomial{d, T},
                                    mapping::Union{Nothing, OMF{T}, ConditionalMapping{d, <:Any, T}}
                        ) where {d, T<:Number}
        new{d, T, typeof(mapping)}(B, Φ, mapping)
    end
end

@inline _of_same_PSD(a::PSDModelPolynomial{<:Any, T}, B::AbstractMatrix{T}) where {T<:Number} =
                PSDModelPolynomial(Hermitian(B), a.Φ, a.mapping)

@inline _of_same_PSD(a::PSDModelPolynomial{<:Any, T}, B::AbstractMatrix{T}, Φ::FMTensorPolynomial{d, T}) where {d, T<:Number} =
                PSDModelPolynomial(Hermitian(B), Φ, a.mapping)

@inline _tensorizer(a::PSDModelPolynomial) = a.Φ.ten

@inline _remove_mapping(a::PSDModelPolynomial{d, T, <:ConditionalMapping{d, <:Any, T}}) where {d, T<:Number} = 
    PSDModelPolynomial(a.B, a.Φ)

@inline _add_mapping(a::PSDModelPolynomial{d, T, Nothing}, mapping::ConditionalMapping{d, <:Any, T}) where {d, T<:Number} = 
    PSDModelPolynomial(a.B, a.Φ, mapping)

## Pretty printing
function Base.show(io::IO, a::PSDModelPolynomial{d, T, S}) where {d, T, S}
    println(io, "PSDModelPolynomial{d=$d, T=$T, S=$S}")
    println(io, "   matrix size: ", size(a.B))
    println(io, "   Φ: ", a.Φ)
end

domain_interval(a::PSDModelPolynomial{d, T}, k::Int) where {d, T<:Number} = begin
    @assert 1 ≤ k ≤ d
    return domain_interval(a.Φ, k)
end

function create_proposal(
        a::PSDModelPolynomial{d, T}, 
        index::Vector{Int}
    ) where {d, T<:Number}
    create_proposal(a, index, ones(T, size(a.B,1)+1))
end

function create_proposal(
        a::PSDModelPolynomial{d, T}, 
        index::Vector{Int},
        vec::Vector{T}
    ) where {d, T<:Number}
    B = ones(T, size(a.B,1)+1, size(a.B,2)+1)
    B[1:end-1, 1:end-1] = a.B
    B[end, :] = vec
    B[:, end] = vec
    b = _of_same_PSD(a, Hermitian(B), 
                add_index(a.Φ, index))
    return b
end

function marginalize(a::PSDModelPolynomial{<:Any, T}, dim::Int; kwargs...) where {T<:Number}
    # if the space is Legendre, the orthonormal measure is Lebesgue.
    if typeof(a.Φ.space.spaces[dim]) isa Jacobi &&
            a.Φ.space.spaces[dim].a == a.Φ.space.spaces[dim].b == 0.0
        return marginalize_orth_measure(a, dim)
    end
    return marginalize(a, dim, x->1.0; kwargs...)
end
function marginalize(a::PSDModelPolynomial{d, T, S}, dim::Int,
                     measure::Function; domain=nothing, kwargs...) where {d, T<:Number, S}
    @assert 1 ≤ dim ≤ d

    mapped_measure = if S === Nothing
        measure
    else
        measure ∘ (ξ -> x(a.mapping, ξ))
    end

    M = if domain===nothing
        calculate_M_quadrature(a.Φ, dim, mapped_measure; kwargs...)
    else
        calculate_M_quadrature(a.Φ, dim, mapped_measure, domain; kwargs...)
    end
    if d-1 == 0  ## no dimension left
        return tr(M.*a.B)
    end

    @inline δ(i::Int, j::Int) = i == j ? 1 : 0
    @inline comp_ind(x,y) = mapreduce(k->k<dim ? δ(x[k], y[k]) : δ(x[k], y[k+1]), *, 1:(d-1))
    new_Φ = reduce_dim(a.Φ, dim)
    P = spzeros(T, new_Φ.N, a.Φ.N)
    for i=1:new_Φ.N
        for j=1:a.Φ.N
            P[i, j] = comp_ind(σ_inv(new_Φ, i), σ_inv(a.Φ, j))
        end
    end

    B = P * (M .* a.B) * P'
    return _of_same_PSD(a, Hermitian(Matrix(B)), new_Φ)
end

marginalize(a::PSDModelPolynomial{d}; kwargs...) where {d} = marginalize(a, collect(1:d); kwargs...)
marginalize(a::PSDModelPolynomial{<:Any, T}, dims::Vector{Int}; kwargs...) where {T<:Number} = marginalize(a, dims, x->1.0; kwargs...)
function marginalize(a::PSDModelPolynomial{d, T}, dims::Vector{Int},
                    measure::Function; kwargs...) where {d, T<:Number}
    @assert 1 ≤ minimum(dims) ≤ maximum(dims) ≤ d
    dims = sort(dims)
    for dim in reverse(dims) ## reverse order to avoid changing the indices
        a = marginalize(a, dim, measure; kwargs...)
    end
    return a
end

function integrate(a::PSDModelPolynomial, dim::Int, p::Function, domain::Domain)
    marginalize(a, dim, p, domain=domain)
end

function compile(a::PSDModelPolynomial{d, T, S}) where {d, T<:Number, S}
    Sym.@variables x[1:d]
    poly = a(x)
    # return eval(Sym.build_function(poly, x))
    ret_f = Sym.build_function(poly, x, expression=Val{false})
    Base.remove_linenums!(ret_f)
    return ret_f
end


"""
    function integral(a::PSDModelPolynomial{d, T, S}, dim::Int; C=nothing)

Integrate the model along a given dimension. The integration constant C gives
the x value of where it should start. If C is not given, it is assumed to be
the beginning of the interval.
"""
function integral(a::PSDModelPolynomial{d, T, S}, dim::Int; C=nothing) where {d, T<:Number, S<:Union{Nothing, OMF{T}}}
    @assert 1 ≤ dim ≤ d
    if C === nothing
        # use the left endpoint of the original domain,
        # even if there is a mapping.
        C = leftendpoint(a.Φ.space.spaces[dim].domain)
    end
    M = SquaredPolynomialMatrix(a.Φ, Int[dim]; C=C)
    return PolynomialTraceModel(a.B, M, a.mapping)
end
integral(_::PSDModelPolynomial{<:Any, T, S}, _::Int) where {T<:Number, S<:ConditionalMapping} = throw(error("An SoS function with a mapping cannot be integrated."))

function compiled_integral(a::PSDModelPolynomial{d, T, S}, dim::Int; C=nothing) where {d, T<:Number, S}
    @assert 1 ≤ dim ≤ d
    M = integral(a, dim; C = C)
    Sym.@variables x[1:d]
    poly = M(x)
    comp_int_poly = Sym.build_function(poly, x, expression=Val{false}, 
                                linenumbers=false, skipzeros=true
                                )
    Base.remove_linenums!(comp_int_poly)
    return comp_int_poly
end

function tensorize(a::PSDModelPolynomial{d, T}, b::PSDModelPolynomial{d, T}) where {d, T<:Number}
    Φ_new = tensorize(a.Φ, b.Φ)
    # B_new = zeros(T, Φ_new.N, Φ_new.N)
    B_new = kron(b.B, a.B)
    return PSDModelPolynomial(Hermitian(B_new), Φ_new)
end

moment_tensor(a::PSDModelPolynomial{d, T}) where {d, T<:Number} = moment_tensor(a, moment_tensor(a.Φ))
function moment_tensor(a::PSDModelPolynomial{d, T}, M::AbstractMatrix{<:AbstractArray{T, d}}) where {d, T<:Number}
    ret = zeros(T, size(M[1,1]))
    for i=1:size(M,1), j=i+1:size(M,2)
        @inbounds ret .+= 2.0 * M[i,j] * a.B[i,j]
    end
    for i=1:size(M,1)
        @inbounds ret .+= M[i,i] * a.B[i,i]
    end
    return ret
end

function get_semialgebraic_domain_constraints(a::PSDModelPolynomial{d, T}) where {d, T<:Number}
    D_list = Vector{Vector{SparseMatrixCSC{T}}}(undef, d)
    # D_list = Vector{Vector{Matrix{T}}}(undef, d)
    coef_list = Vector{Vector{T}}(undef, d)
    for i=1:d
        @assert a.Φ.space.spaces[i].a == 0.0
        @assert a.Φ.space.spaces[i].b == 0.0

        q_1 = Fun(a.Φ.space.spaces[i], [1.0])
        q_2 = Fun(a.Φ.space.spaces[i], [0.0, 1.0])
        D = mat_D(a.Φ, [q_1, q_2], i)

        D_rem_col = [d[:, vec(mapslices(col -> any(col .!= 0), d, dims = 1))] for d in D]
        D_list[i] = D_rem_col
        coef_list[i] = [1.0, -1.0]
    end
    return D_list, coef_list
end

import LinearAlgebra
LinearAlgebra.normalize(a::PSDModelPolynomial{d, <:Number}, measure::Function; kwargs...) where {d} = _of_same_PSD(a, a.B * (1/marginalize(a, collect(1:d), measure; kwargs...)))
LinearAlgebra.normalize(a::PSDModelPolynomial{d, <:Number}; kwargs...) where {d} = _of_same_PSD(a, a.B * (1/marginalize(a, collect(1:d); kwargs...)))
normalize_orth_measure(a::PSDModelPolynomial{<:Any, <:Number}) = _of_same_PSD(a, a.B * (1/tr(a.B)))
normalize_orth_measure!(a::PSDModelPolynomial{<:Any, T}) where {T<:Number} = a.B .= a.B * (1/tr(a.B))
LinearAlgebra.normalize!(a::PSDModelPolynomial{d, T}; kwargs...) where {d, T<:Number} = a.B .= a.B * (1/marginalize(a, collect(1:d); kwargs...))
LinearAlgebra.normalize!(a::PSDModelPolynomial{d, T}, measure::Function; kwargs...) where {d, T<:Number} = a.B .= a.B * (1/marginalize(a, collect(1:d), measure; kwargs...))