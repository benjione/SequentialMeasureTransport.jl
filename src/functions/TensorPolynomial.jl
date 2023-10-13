"""
A feature map of tensorization of polynimials, where ``\\sigma``
is the function that maps a multiindex to a coefficient index
in the tensorization.
"""
struct FMTensorPolynomial{d, T, S<:Tensorizer, tsp<:TensorSpace} <: TensorFunction{d, T, S}
    space::tsp
    normal_factor::Vector{Vector{T}} # Normalization factor for polynomials
    N::Int                  # order of feature map
    ten::S                  # tensorizer
    highest_order::Int      # max(order(p_i)) for all p_i in dimensions
    function FMTensorPolynomial{d, T}(space::tsp,
                    normal_factor::Vector{Vector{T}}, 
                    N::Int,
                    ten::S,
                    highest_order::Int) where {d, T, S<:Tensorizer, tsp<:TensorSpace}
        return new{d, T, S, tsp}(space, normal_factor, N, ten, highest_order)
    end
end

dimensions(::FMTensorPolynomial{d}) where {d} = d
domain_interval(p::FMTensorPolynomial, dim::Int) = begin
    return (leftendpoint(p.space.spaces[dim].domain),
    rightendpoint(p.space.spaces[dim].domain))
end

@inline σ(p::FMTensorPolynomial{<:Any, <:Any, S}, i) where {S<:Tensorizer} = σ(p.ten, i)
@inline σ_inv(p::FMTensorPolynomial{<:Any, <:Any, S}, i) where {S<:Tensorizer} = σ_inv(p.ten, i)

## Pretty printing
Base.show(io::IO, p::FMTensorPolynomial{d, T, S, tsp}) where {d, T, S<:Tensorizer, tsp<:TensorSpace} = begin
    println(io, "FMTensorPolynomial{d=$d, T=$T, ...}")
    println(io, "   space: ", p.space)
    println(io, "   highest order: ", p.highest_order)
    println(io, "   N: ", p.N)
end

function add_index(p::FMTensorPolynomial{d, T}, index::Vector{Int}) where {d, T}
    ten = deepcopy(p.ten)
    add_index!(ten, index)
    normalization_factor = set_normalization_factors(T, p.space, highest_order(ten))
    return FMTensorPolynomial{d, T}(p.space, normalization_factor, p.N+1, ten, highest_order(ten)-1)
end

function reduce_dim(p::FMTensorPolynomial{d, T}, dim::Int) where {d, T}
    ten_new = reduce_dim(p.ten, dim)
    if d-1 == 0
        return FMTensorPolynomial{0, T}(TensorSpace(ConstantSpace()), Vector{T}[], 1, ten_new, 0)
    end
    triv_new_N = max_N(ten_new)
    new_N = 0
    # x is new index, y is old index
    @inline comp_ind(x,y) = mapreduce(i->i<dim ? x[i]==y[i] : x[i]==y[i+1], *, 1:d-1)
    for i=1:triv_new_N
        for j=1:p.N
            if comp_ind(σ_inv(ten_new, i), σ_inv(p.ten, j))
                new_N += 1
                break
            end
        end
    end
    space = TensorSpace([p.space.spaces[i] for i=1:d if i≠dim]...)
    norm_factors = p.normal_factor[setdiff(1:d, dim)]
    return FMTensorPolynomial{d-1, T}(space, norm_factors, new_N, ten_new, p.highest_order)
end

### Constructors
trivial_TensorPolynomial(T::Type{<:Number}, sp::Space, N::Int) = trivial_TensorPolynomial(T, TensorSpace(sp), N)
function trivial_TensorPolynomial(T::Type{<:Number},
                                space::TensorSpace, 
                                N::Int)
    d = length(space.spaces)
    ten = TrivialTensorizer(d, N)
    high_order = highest_order(ten)-1
    normal_factor = set_normalization_factors(T, space, high_order)
    return FMTensorPolynomial{d, T}(space, normal_factor, N, ten, high_order)
end

downwardClosed_Polynomial(T::Type{<:Number}, sp::Space, max_order::Int) = trivial_TensorPolynomial(T, TensorSpace(sp), max_order)
function downwardClosed_Polynomial(T::Type{<:Number},
                                    space::TensorSpace, 
                                   max_order::Int)
    d = length(space.spaces)
    ten = DownwardClosedTensorizer(d, max_order)
    normal_factor = set_normalization_factors(T, space, max_order)
    N = max_N(ten)
    return FMTensorPolynomial{d, T}(space, normal_factor, N, ten, max_order)
end

function add_order(p::FMTensorPolynomial{d, T}, dim::Int) where {d, T<:Number}
    ten_new = add_order(p.ten, dim)
    triv_new_N = max_N(ten_new)
    ## work with trivial new N now, maybe other strategy later
    normal_vecs = set_normalization_factors(T, p.space, highest_order(ten_new)-1)
    return FMTensorPolynomial{d, T}(p.space, normal_vecs, triv_new_N, ten_new, p.highest_order+1)
end

(p::FMTensorPolynomial{d, T})(x::T) where {d, T} = p(T[x])
function (p::FMTensorPolynomial{d, T})(x::AbstractVector{T}) where {d, T}
    @assert length(x) == d
    A = zeros(T, p.highest_order+1, d)
    poly(k,i) = Fun(p.space.spaces[i], T[zeros(T, k);p.normal_factor[i][k+1]])(x[i])
    map!(t->poly(t...), A, collect(Iterators.product(0:p.highest_order, 1:d)))

    @inline Ψ(k) = mapreduce(j->A[k[j], j], *, 1:d)
    map(i -> Ψ(σ_inv(p, i)), 1:p.N)
end


_eval(p::FMTensorPolynomial{d, T}, x::T, ignore_dim::Vector{Int}) where {d, T} = _eval(p, T[x], ignore_dim)
function _eval(p::FMTensorPolynomial{d, T, S, tsp}, 
               x::AbstractVector{T},
               ignore_dim::Vector{Int}) where {d, T, S, tsp<:TensorSpace}
    @assert length(x) == d
    iter_dim = setdiff(1:d, ignore_dim)
    A = zeros(T, p.highest_order+1, d)
    poly(k::Int,i::Int) = Fun(p.space.spaces[i], T[zeros(T, k);p.normal_factor[i][k+1]])(x[i]::T)::T
    map!(t->poly(t...), A, collect(Iterators.product(0:p.highest_order, 1:d)))
    @inline Ψ(k) = mapreduce(j->A[k[j], j], *, iter_dim, init=one(T))
    return map(i -> Ψ(σ_inv(p, i)), 1:p.N)
end




"""
Calculates M_{σ(i)σ(j)} = \\int measure(x) \\phi_{i_dim}(x) \\phi_{j_dim}(x) dx
using Gauss-Legendre quadrature.
"""
function calculate_M_quadrature(p::FMTensorPolynomial{d, T}, 
    dim::Int, 
    measure::Function;
    kwargs...
) where {d, T<:Number}
    calculate_M_quadrature(p,dim,measure,domain_interval(p, dim); kwargs...)
end

"""
Remark:
This also works for Orthogonal Mapped functions, when the
domain endpoints are those of the non mapped functions and the measure is not
dependent on x (Lebesgue measure).

TODO: What is for non Lebesgue measure?
"""
function calculate_M_quadrature(p::FMTensorPolynomial{d, T}, 
                                dim::Int, 
                                measure::Function,
                                domain_endpoints::Tuple{<:Number, <:Number};
                                amount_quadrature_points=nothing
                        ) where {d, T<:Number}
    M = zeros(T, p.N, p.N)
    qaudr_order = if amount_quadrature_points === nothing
        p.highest_order+2
    else
        amount_quadrature_points
    end
    x, w = gausslegendre(qaudr_order) # rule is exact for order 2n-1
    ## scale Gauss quadrature to domain
    l, r = domain_endpoints
    x .*= ((r - l)/2)
    x .+= ((r + l)/2)
    corr = ((r - l)/2)

    A = T[Fun(p.space.spaces[dim], [zeros(T, k);p.normal_factor[dim][k+1]])(x_i) for k=0:p.highest_order, x_i in x]
    meas_vec = measure.(x)

    integrate(i,j) = corr * dot(w, meas_vec.*A[i,:].*A[j,:])
    for i=1:p.N
        for j=i:p.N
            M[i,j] = integrate(σ_inv(p, i)[dim],σ_inv(p, j)[dim])
        end
    end
    return Symmetric(M)
end

set_normalization_factors(T::Type{<:Number}, ps::TensorSpace, high_order::Int) = set_normalization_factors(T, ps, high_order*ones(Int, length(ps.spaces)))
function set_normalization_factors(T::Type{<:Number}, poly_spaces::TensorSpace, highest_orders::Vector{Int})
    d = length(poly_spaces.spaces)
    @assert d == length(highest_orders)
    normal_factor = Vector{T}[]
    for i=1:d
        norm_vec = set_normalization_factors(T, poly_spaces.spaces[i], highest_orders[i])
        push!(normal_factor, norm_vec)
    end
    return normal_factor
end

function set_normalization_factors(T::Type{<:Number}, poly_space::Space, highest_order::Int)
    normal_factor = zeros(T, highest_order+1)
    for i=0:highest_order
        normal_factor[i+1] = norm_func(T, poly_space, i)
    end
    return normal_factor
end

# default volume of Chebyshev is 2.0
function norm_func(T::Type{<:Number}, sp::Chebyshev, n)
    vol_change = sqrt(2.0/volume(sp.domain))
    if n==0
        return T(vol_change/sqrt(T(π)))
    else
        return T(sqrt(2.0)*vol_change/sqrt(T(π)))
    end
end
function norm_func(T::Type{<:Number}, sp::Jacobi, n)
    vol_change = sqrt(2.0/volume(sp.domain))
    if sp.a == sp.b == 0 # Legendre
        return T(sqrt((2n+1)/2) * vol_change)
    else
        @error "Not implemented"
    end
end
norm_func(T::Type{<:Number}, sp::Hermite, n) = begin
    @assert sp.L == 1.0
    return T(sqrt(1/(sqrt(T(π)) * 2^n * factorial(n))))
end
norm_func(T::Type{<:Number}, ::Any, n) = @error "Not implemented"