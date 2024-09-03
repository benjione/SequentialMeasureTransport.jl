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

function reduce_dim(p::FMTensorPolynomial{d, T}, dims::Vector{Int}) where {d, T}
    p_red = p
    for dim in sort(dims, rev=true)
        p_red = reduce_dim(p_red, dim)
    end
    return p_red
end

function tensorize(p1::FMTensorPolynomial{d1, T}, p2::FMTensorPolynomial{d2, T}) where {d1, d2, T}
    ten_new = tensorize(p1.ten, p2.ten)
    space = TensorSpace(p1.space.spaces..., p2.space.spaces...)
    new_highest_order = max(highest_order(p1.ten), highest_order(p2.ten))
    normalization_factor = set_normalization_factors(T, space, new_highest_order)
    return FMTensorPolynomial{d1+d2, T}(space, normalization_factor, p1.N*p2.N, ten_new, new_highest_order)
end

function permute_indices(p::FMTensorPolynomial{d, T}, perm::Vector{Int}) where {d, T}
    @assert length(perm) == d
    @assert length(unique(perm)) == d
    ten_new = permute_indices(p.ten, perm)
    norm_factors = p.normal_factor[perm]
    space = reduce((x,y)->x ⊗ y, p.space.spaces[perm])
    if d==1
        return FMTensorPolynomial{1, T}(TensorSpace(space), norm_factors, p.N, ten_new, p.highest_order)
    end
    return FMTensorPolynomial{d, T}(space, norm_factors, p.N, ten_new, p.highest_order)
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

downwardClosed_Polynomial(T::Type{<:Number}, 
                        sp::Space, 
                        max_order::Int;
                        kwargs...) = downwardClosed_Polynomial(T, TensorSpace(sp), max_order)
function downwardClosed_Polynomial(T::Type{<:Number},
                                    space::TensorSpace, 
                                   max_order::Int;
                                   max_Φ_size=nothing)
    d = length(space.spaces)
    ten = DownwardClosedTensorizer(d, max_order; max_Φ_size=max_Φ_size)
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
function (p::FMTensorPolynomial{d, T})(x::AbstractVector{T}) where {d, T<:Number}
    @assert length(x) == d
    # A = Array{T}(undef, p.highest_order+1, d)
    # poly(k,i) = Fun(p.space.spaces[i], T[zeros(T, k);p.normal_factor[i][k+1]])(x[i])
    # map!(t->poly(t...), A, collect(Iterators.product(0:p.highest_order, 1:d)))

    A = Array{T}(undef, p.highest_order+1, d)
    @inbounds for i=1:d
        A[:, i] = ApproxFun.ApproxFunOrthogonalPolynomials.forwardrecurrence(T, p.space.spaces[i], 0:p.highest_order, ApproxFun.tocanonical(p.space.spaces[i], x[i]))
        A[:, i] .*= p.normal_factor[i]
    end

    @inline Ψ(k) = mapreduce(j->A[k[j], j], *, 1:d)
    map(i -> Ψ(σ_inv(p, i)), 1:p.N)
end

function (p::FMTensorPolynomial{d, T})(x::AbstractVector{T2}) where {d, T<:Number, T2}
    @assert length(x) == d
    A = zeros(T2, p.highest_order+1, d)
    poly(k,i) = begin
        l = ApproxFun.leftendpoint(p.space.spaces[i].domain)
        r = ApproxFun.rightendpoint(p.space.spaces[i].domain)
        ApproxFun.clenshaw(p.space.spaces[i], T[zeros(T, k);p.normal_factor[i][k+1]], (x[i]-l)/(r-l) * 2.0 - 1.0)
    end
    map!(t->poly(t...), A, collect(Iterators.product(0:p.highest_order, 1:d)))

    @inline Ψ(k) = mapreduce(j->A[k[j], j], *, 1:d)
    map(i -> Ψ(σ_inv(p, i)), 1:p.N)
end

_eval(p::FMTensorPolynomial{d, T}, x::T, ignore_dim::Vector{Int}) where {d, T} = _eval(p, T[x], ignore_dim)
function _eval(p::FMTensorPolynomial{d, T, S, tsp}, 
               x::AbstractVector{T},
               ignore_dim::Vector{Int}) where {d, T<:Number, S, tsp<:TensorSpace}
    @assert length(x) == d
    iter_dim = setdiff(1:d, ignore_dim)
    # A = zeros(T, p.highest_order+1, d)
    # poly(k::Int,i::Int) = Fun(p.space.spaces[i], T[zeros(T, k);p.normal_factor[i][k+1]])(x[i]::T)::T
    # map!(t->poly(t...), A, collect(Iterators.product(0:p.highest_order, 1:d)))
    
    A = Array{T}(undef, p.highest_order+1, d)
    @inbounds for i=1:d
        if i in ignore_dim
            # A[:, i] .= 1.0
            continue
        end
        A[:, i] = ApproxFun.ApproxFunOrthogonalPolynomials.forwardrecurrence(T, p.space.spaces[i], 0:p.highest_order, ApproxFun.tocanonical(p.space.spaces[i], x[i]))
        A[:, i] .*= p.normal_factor[i]
    end
    
    @inline Ψ(k) = mapreduce(j->A[k[j], j], *, iter_dim, init=one(T))
    return map(i -> Ψ(σ_inv(p, i)), 1:p.N)
end

function _eval(p::FMTensorPolynomial{d, T, S, tsp}, 
               x::AbstractVector{T2},
               ignore_dim::Vector{Int}) where {d, T<:Number, T2<:Number, S, tsp<:TensorSpace}
    @assert length(x) == d
    iter_dim = setdiff(1:d, ignore_dim)
    A = zeros(T2, p.highest_order+1, d)
    poly(k,i) = begin
        l = ApproxFun.leftendpoint(p.space.spaces[i].domain)
        r = ApproxFun.rightendpoint(p.space.spaces[i].domain)
        ApproxFun.clenshaw(p.space.spaces[i], T[zeros(T, k);p.normal_factor[i][k+1]], (x[i]-l)/(r-l) * 2.0 - 1.0)
    end
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

    _integrate(i,j) = corr * dot(w, meas_vec.*A[i,:].*A[j,:])
    for i=1:p.N
        for j=i:p.N
            M[i,j] = _integrate(σ_inv(p, i)[dim],σ_inv(p, j)[dim])
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

function moment_matrix_legendre(degree, norm_fac_vec)
    mom = spzeros(degree+1, degree+1)
    mom[1, 1] = 1.0
    mom[2, 2] = 1.0
    for i=3:degree+1
        j = i-2
        mom[i, :] = (2*j+1)/(j+1) * circshift(mom[i-1, :], 1) - j/(j+1) * mom[i-2, :]
    end
    for (i, n_const) in enumerate(norm_fac_vec)
        mom[i, :] .*= n_const
    end
    return mom
end

using DSP
function moment_tensor(Φ::FMTensorPolynomial{d, T}; threading=true) where {d, T}
    for i=1:d
        @assert Φ.space.spaces[i] isa Jacobi
        @assert Φ.space.spaces[i].a == Φ.space.spaces[i].b == 0
    end
    legendre_mom = moment_matrix_legendre(Φ.highest_order, Φ.normal_factor[1])
    mom = Vector{SparseVector{T}}(undef, Φ.N)
    @_condusethreads threading for i=1:Φ.N
        indices = σ_inv(Φ, i)
        mom_tensor = foldl((x, j)->kron(x, legendre_mom[indices[j], :]), 1:d, init=one(T))
        # mom_tensor = reshape(mom_tensor, fill(Φ.highest_order+1, d)...)
        mom[i] = mom_tensor
    end
    mom_matrix = Matrix{Array{T, d}}(undef, Φ.N, Φ.N)
    @_condusethreads threading for i=1:Φ.N
        for j=i:Φ.N
            res = conv(reshape(mom[i], fill(Φ.highest_order+1, d)...), reshape(mom[j], fill(Φ.highest_order+1, d)...))
            # res = conv(mom[i], mom[j])
            res[abs.(res) .< 1e-12] .= 0.0
            mom_matrix[i, j] = res
            mom_matrix[j, i] = res
        end
    end
    return mom_matrix
end


function mat_D(Φ::FMTensorPolynomial{d, T}, q_list, dim::Int) where {d, T}
    for q in q_list
        @assert q.space.domain.left == Φ.space.spaces[dim].domain.left
        @assert q.space.domain.right == Φ.space.spaces[dim].domain.right
    end
    @inline δ(i::Int, j::Int) = i == j ? 1 : 0
    @inline δ(i, j, not_dim) = mapreduce(k->k==not_dim ? true : i[k]==j[k],*, 1:d)
    D_list = [spzeros(T, Φ.N, Φ.N) for _=1:length(q_list)]
    # j_ignore = []
    highest_order_dim = maximum([σ_inv(Φ, j)[dim] for j=1:Φ.N])
    for i=1:Φ.N
        ind_i = σ_inv(Φ, i)
        Φ_i = Fun(Φ.space.spaces[dim], [zeros(ind_i[dim]-1); Φ.normal_factor[dim][ind_i[dim]]])
        Φ_new_list = [Φ_i * q for q in q_list]
        # Φ_new = Φ_i * q
        out_of_order = false
        for Φ_new in Φ_new_list
            new_vec = Φ_new.coefficients
            for j=1:length(new_vec)
                ind_i_tmp = (ind_i[1:dim-1]..., j, ind_i[dim+1:end]...)
                if !check_in_tensorizer(Φ.ten, ind_i_tmp)
                    out_of_order = true
                    # break
                end
            end
        end
        if out_of_order
            continue
        end
        for (D, Φ_new) in zip(D_list, Φ_new_list)
            new_vec = Φ_new.coefficients
            # if Φ.highest_order+1 < length(new_vec)
            #     continue
            # end
            new_vec = new_vec[1:minimum([Φ.highest_order+1, length(new_vec)])]
            new_vec ./= Φ.normal_factor[dim][1:length(new_vec)]
            for j=1:Φ.N
                ind_j = σ_inv(Φ, j)
                if δ(ind_i, ind_j, dim)
                    if length(new_vec) ≥ ind_j[dim] && new_vec[ind_j[dim]] ≠ 0
                        D[j, i] = new_vec[ind_j[dim]]
                    # elseif length(new_vec) < ind_j[dim]
                    #     push!(j_ignore, j)
                    end
                end
            end
        end
    end
    # for D in D_list
    #     for j in j_ignore
    #         D[j, :] .= 0.0
    #     end
    # end
    return D_list
end