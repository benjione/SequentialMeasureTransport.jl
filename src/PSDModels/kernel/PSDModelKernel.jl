# kwargs definition of PSDModelKernel
const _PSDModelKernel_kwargs =
        (:use_view, )


struct PSDModelKernel{T<:Number} <: PSDModel{T}
    B::Hermitian{Float64, <:AbstractMatrix{Float64}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    k::Kernel                               # k(x, y) is the kernel function
    X::PSDDataVector{T}                     # X is the set of points for the feature map
    function PSDModelKernel(B::Hermitian{Float64, <:AbstractMatrix{Float64}}, 
                        k::Kernel, 
                        X::PSDDataVector{T};
                        use_view=false
                    ) where {T<:Number}
            
        X = if use_view
            @view X[1:end] # protect from appending
        else
            deepcopy(X)    # protect from further changes
        end
        new{T}(B, k, X)
    end
end

function PSDModelKernel(k::Kernel, X::PSDDataVector{T}; kwargs...) where {T<:Number}
    B = rand(Manifolds.SymmetricPositiveDefinite(length(X)))
    PSDModelKernel(Hermitian(B), k, X; kwargs...)
end

@inline _of_same_PSD(a::PSDModelKernel{T}, B::AbstractMatrix{T}) where {T<:Number} =
                                PSDModelKernel(Hermitian(B), a.k, a.X)

dimension(a::PSDModelKernel) = length(a.X[1])

"""
Φ(a::PSDModelKernel, x::PSDdata{T}) where {T<:Number}

Returns the feature map of the PSD model at x.
"""
function Φ(a::PSDModelKernel, x::PSDdata{T}) where {T<:Number}
    return a.k.(Ref(x), a.X)
end

function PSDModel_gradient_descent(
        X::PSDDataVector{T},
        Y::Vector{T},
        k::Kernel;
        λ_1=1e-8,
        trace=false,
        B0=nothing,
        kwargs...
    ) where {T<:Number}
    N = length(X)
    A0 = if B0===nothing
        diagm(ones(N))
    else
        B0
    end
    model = PSDModelKernel(Hermitian(A0), k, X; 
                  _filter_kwargs(kwargs, _PSDModelKernel_kwargs)...)
    fit!(model, X, Y; trace=trace, λ_1=λ_1, kwargs...)
    return model
end

function PSDModel_direct(
        X::PSDDataVector{T}, 
        Y::Vector{T}, 
        k::Kernel;
        regularize_kernel=true,
        cond_thresh=1e4,
        λ_1=1e-3,
        λ_2=1e-8,
        trace=false,
        kwargs...
    ) where {T<:Number}
    K = kernelmatrix(k, X)
    K = Hermitian(K)

    trace && @show cond(K)

    if regularize_kernel && (cond(K) > cond_thresh)
        K += λ_1 * I
        if trace
            @show "Kernel has been regularized"
            @show λ_1
            @show cond(K)
        end
    end

    @assert isposdef(K)

    V = cholesky(K)
    V_inv = inv(V)
    V2 = Matrix(V)

    A = Hermitian(spdiagm(Y.+1e-8))
    evals, evecs = eigen((V2 * A * V2' + λ_1 * I))
    evals = max.(evals, Ref(0.0))
    B = Hermitian((evecs * Diagonal(evals) * evecs'))
    B = Hermitian((V_inv' * B * V_inv))

    # # project B onto the PSD cone, just in case
    # B, _ = prox(IndPSD(), B)

    return PSDModelKernel(B, k, X; 
                    _filter_kwargs(kwargs, _PSDModelKernel_kwargs)...)
end


function set_coefficients(a::PSDModelKernel{T}, B::Hermitian{T}) where {T<:Number}
    return PSDModel{T}(B, a.k, a.X)
end

"""
add_support(a::PSDModel{T}, X::PSDdata{T}) where {T<:Number}

Returns a PSD model with added support points, where the model still gives
the same results as before (extension of the matrix initialized with zeros).
"""
function add_support(a::PSDModelKernel{T}, X::PSDDataVector{T}) where {T<:Number}
    new_S = vcat(a.X, X)
    B = Hermitian(vcat(
                    hcat(a.B, zeros(Float64, length(X), length(X))), 
                    zeros(Float64, length(X), length(a.X)+length(X))
                 )
        )
    return PSDModelKernel(B, a.k, new_S)
end

"""
Marginalize for RBF kernel, where the kernel is defined as
    k(x, y) = σ^2 * exp(-||x - y||^2 / (2l^2))
"""
function marginalize(a::PSDModelKernel{T}, dim::Int; σ=1.0) where {T<:Number}

    l = 1.0
    if typeof(a.k) <: KernelFunctions.TransformedKernel
        @assert typeof(a.k.kernel) <: KernelFunctions.SqExponentialKernel
        @assert typeof(a.k.transform) <: KernelFunctions.ScaleTransform
        l = 1/a.k.transform.s[1]
    else
        @assert typeof(a.k) <: KernelFunctions.SqExponentialKernel
    end

    X_dim = [x[dim] for x in a.X]

    ## the result is \exp(-||x_1 - x_2||^2 / (4.0 l^2)) * l (\sqrt(\pi))

    K = ((X_dim .- X_dim')).^2
    K = exp.(-K ./ (4.0*l^2)) * σ^4 * (sqrt(π)) * l
    B = a.B .* K
    _d = length(a.X[1])
    if _d == 1
        return sum(B)
    end
    X_new = [x[1:dim-1] ∪ x[dim+1:_d] for x in a.X]
    return PSDModelKernel(Hermitian(B), a.k, X_new)
end

function marginalize(a::PSDModelKernel{T}, dims::Vector{Int}; σ=1.0) where {T<:Number}
    d = length(a.X[1])
    @assert 1 ≤ minimum(dims) ≤ maximum(dims) ≤ d
    dims = sort(dims)
    for dim in reverse(dims) ## reverse order to avoid changing the indices
        a = marginalize(a, dim; σ=σ)
    end
    return a
end

normalize!(a::PSDModelKernel{T}) where {T<:Number} = begin
    d = length(a.X[1])
    c = marginalize(a, collect(1:d))
    a.B .= a.B / c
end

normalize(a::PSDModelKernel{T}) where {T<:Number} = begin
    d = length(a.X[1])
    c = marginalize(a, collect(1:d))
    return PSDModelKernel(a.B / c, a.k, a.X)
end

function integration_matrix(a::PSDModelKernel{T}) where {T<:Number}
    l = 1.0
    if typeof(a.k) <: KernelFunctions.TransformedKernel
        @assert typeof(a.k.kernel) <: KernelFunctions.SqExponentialKernel
        @assert typeof(a.k.transform) <: KernelFunctions.ScaleTransform
        l = 1/a.k.transform.s[1]
    else
        @assert typeof(a.k) <: KernelFunctions.SqExponentialKernel
    end
    d = length(a.X[1])

    ## the result is \exp(-||x_1 - x_2||^2 / (4.0 l^2)) * l (\sqrt(\pi))
    new_k = KernelFunctions.TransformedKernel(
            KernelFunctions.SqExponentialKernel(),
            KernelFunctions.ScaleTransform(0.5/l))
    K = kernelmatrix(new_k, a.X)
    # K = ((a.X .- a.X')).^2
    # K = exp.(-K ./ (4.0*l^2)) * σ^4 * (sqrt(π)) * l
    K .= K .* ((sqrt(π)) * l)^d
    return K
end


function integral(a::PSDModelKernel{T}, dim::Int) where {T<:Number}
    return KernelTraceModel(a.X, a.B, a.k, dim)
end

function permute_indices(a::PSDModelKernel{T}, perm::Vector{Int}) where {T<:Number}
    return PSDModelKernel(a.B, a.k, [x[perm] for x in a.X])
end