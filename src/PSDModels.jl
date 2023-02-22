module PSDModels

using LinearAlgebra, SparseArrays
using KernelFunctions: Kernel, kernelmatrix
using ProximalOperators: IndPSD, prox, prox!
using DomainSets
using FastGaussQuadrature: gausslegendre
import ForwardDiff as FD
import ProximalAlgorithms
import Base

include("utils.jl")
include("optimization.jl")

export PSDModel
export fit!, minimize!
export gradient, integral

# for working with 1D and nD data
const PSDdata{T} = Union{T, Vector{T}} where {T<:Number}
const PSDDataVector{T} = Union{Vector{T}, Vector{Vector{T}}} where {T<:Number}

# kwargs definition of PSDModel
const _PSDModel_kwargs =
        (:use_view, )

struct PSDModel{T<:Number}
    B::Hermitian{Float64, Matrix{Float64}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    k::Kernel                               # k(x, y) is the kernel function
    X::PSDDataVector{T}                     # X is the set of points for the feature map
    function PSDModel(B::Hermitian{Float64, Matrix{Float64}}, 
                        k::Kernel, 
                        X::PSDDataVector{T};
                        use_view=false
                    ) where {T<:Number}
            
        X = if use_view
            @view X[1:end] # protect from appending
        else
            copy(X)       # protect from further changes
        end
        new{T}(B, k, X)
    end
end

function PSDModel(k::Kernel, X::PSDDataVector{T}; kwargs...) where {T<:Number}
    B = diagm(ones(Float64, length(X)))
    return PSDModel(Hermitian(B), k, X; 
                    _filter_kwargs(kwargs, _PSDModel_kwargs)...)
end

function PSDModel(
                X::PSDDataVector{T}, 
                Y::Vector{T}, 
                k::Kernel;
                solver=:direct,
                kwargs...
            ) where {T<:Number}
    if solver == :direct
        return PSDModel_direct(X, Y, k; kwargs...)
    elseif solver == :gradient_descent
        return PSDModel_gradient_descent(X, Y, k; kwargs...)
    else
        @error "Solver not implemented"
        return nothing
    end
end

"""
add_support(a::PSDModel{T}, X::PSDdata{T}) where {T<:Number}

Returns a PSD model with added support points, where the model still gives
the same results as before (extension of the matrix initialized with zeros).
"""
function add_support(a::PSDModel{T}, X::PSDdata{T}) where {T<:Number}
    new_S = vcat(a.X, X)
    B = Hermitian(vcat(
                    hcat(a.B, zeros(Float64, length(X), length(X))), 
                    zeros(Float64, length(X), length(a.X)+length(X))
                 )
        )
    return PSDModel(B, a.k, new_S)
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
    K = kernelmatrix(k, X)

    N = length(X)
    
    f_A(i, A::AbstractMatrix) = begin
        v = K[i,:]
        return v' * A * v
    end
    f_A(A::AbstractMatrix) = (1.0/N) * mapreduce(i-> (f_A(i, A) - Y[i])^2, +, 1:N) + λ_1 * tr(A)

    A0 = if B0===nothing
        ones(N,N)
    else
        B0
    end
    solution = optimize_PSD_model(A0, f_A;
                                convex=true,
                                trace=trace,
                                _filter_kwargs(kwargs, 
                                        _optimize_PSD_kwargs,
                                        (:convex, :trace)
                                )...
                            )
    return PSDModel(solution, k, X; 
                    _filter_kwargs(kwargs, _PSDModel_kwargs)...)
end

function PSDModel_direct(
                X::PSDDataVector{T}, 
                Y::Vector{T}, 
                k::Kernel;
                regularize_kernel=true,
                cond_thresh=1e10,
                λ_1=1e-8,
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

    A = Hermitian(spdiagm(Y))
    B = Hermitian((V_inv' * A * V_inv))

    # project B onto the PSD cone, just in case
    B, _ = prox(IndPSD(), B)

    return PSDModel(B, k, X; 
                    _filter_kwargs(kwargs, _PSDModel_kwargs)...)
end

fit!(a::PSDModel, 
        X::PSDDataVector{T}, 
        Y::Vector{T}; 
        kwargs...
    ) where {T<:Number} = fit!(a, X, Y, ones(T, length(X)); kwargs...)
function fit!(a::PSDModel{T}, 
                X::PSDDataVector{T}, 
                Y::Vector{T},
                weights::Vector{T}; 
                λ_1=1e-8,
                trace=false,
                pre_eval=true,
                pre_eval_thresh=5000,
                kwargs...
            ) where {T<:Number}
    N = length(X)

    f_B = if pre_eval && (N < pre_eval_thresh)
        let K = Float64[a.k(x, y) for x in X, y in a.X]
            (i, A::AbstractMatrix) -> begin
                v = K[i,:]
                return v' * A * v
            end
        end
    else
        (i, A::AbstractMatrix) -> begin
            return a(X[i], A)
        end
    end
    f_A(A::AbstractMatrix) = begin
        (1.0/N) * mapreduce(i-> weights[i]*(f_B(i, A) - Y[i])^2, +, 1:N) + λ_1 * tr(A)
    end

    solution = optimize_PSD_model(a.B, f_A;
                                convex=true,
                                trace=trace,
                                _filter_kwargs(kwargs, 
                                        _optimize_PSD_kwargs,
                                        (:convex, :trace)
                                )...)
    set_coefficients!(a, solution)
    return nothing
end

"""
minimize!(a::PSDModel{T}, L::Function, X::PSDDataVector{T}; λ_1=1e-8,
                    trace=false,
                    maxit=5000,
                    tol=1e-6,
                    pre_eval=true,
                    pre_eval_thresh=5000,
                ) where {T<:Number}

Minimizes ``B^* = \\argmin_B L(a_B(x_1), a_B(x_2), ...) + λ_1 tr(B) `` and returns the modified PSDModel with the right matrix B.
"""
function minimize!(a::PSDModel{T}, 
                   L::Function, 
                   X::PSDDataVector{T};
                   λ_1=1e-8,
                   trace=false,
                   pre_eval=true,
                   pre_eval_thresh=5000,
                   kwargs...
            ) where {T<:Number}
    N = length(X)
    f_B = if pre_eval && (N < pre_eval_thresh)
        let K = Float64[a.k(x, y) for x in X, y in a.X]
            (i, A::AbstractMatrix) -> begin
                v = K[i,:]
                return v' * A * v
            end
        end
    else
        (i, A::AbstractMatrix) -> begin
            return a(X[i], A)
        end
    end
    loss(A::AbstractMatrix) = L([f_B(i, A) for i in 1:length(X)]) + λ_1 * tr(A)

    solution = optimize_PSD_model(a.B, loss;
                                trace=trace,
                                _filter_kwargs(kwargs, 
                                        _optimize_PSD_kwargs
                                )...)
    set_coefficients!(a, solution)
end

function (a::PSDModel)(x::PSDdata{T}) where {T<:Number}
    v = a.k.(Ref(x), a.X)
    return v' * a.B * v
end

function (a::PSDModel)(x::PSDdata{T}, B::AbstractMatrix{T}) where {T<:Number}
    v = a.k.(Ref(x), a.X)
    return v' * B * v
end

function set_coefficients!(a::PSDModel{T}, B::Hermitian{T}) where {T<:Number}
    a.B .= B
end

function set_coefficients(a::PSDModel{T}, B::Hermitian{T}) where {T<:Number}
    return PSDModel{T}(B, a.k, a.X)
end

function gradient(a::PSDModel{T}, x::T) where {T<:Number}
    # ∇v = FD.derivative((y)->a.k.(Ref(y), a.X), x)
    # v = a.k.(Ref(x), a.X)
    # return 2 * ∇v' * a.B * v
    
    # ForwardDiff faster than manual implementation
    return FD.derivative(a, x)
end

function parameter_gradient(a::PSDModel{T}, x::T) where {T<:Number}
    v = a.k.(Ref(x), a.X)
    # ∇B = FD.derivative((B)->v' * B * v, a.B)

    ∇B = Matrix{T}(undef, size(a.B)...)
    @inbounds @simd for i in CartesianIndices(a.B)
        ∇B[i] = v[i[1]] * v[i[2]]
    end
    return ∇B
end

function integral(a::PSDModel{T}, χ::Domain; kwargs...) where {T<:Number}
    return integral(a, x->1.0, χ; kwargs...)
end


"""
integral(a::PSDModel{T}, p::Function, χ::Domain; quadrature_method=gausslegendre, amount_quadrature_points=10) where {T<:Number}

returns ``\\int_χ p(x) a(x) dx``. The idea of the implementation is from proposition 4 in [1]. 
The integral is approximated by a quadrature rule. The default quadrature rule is Gauss-Legendre.

[1] U. Marteau-Ferey, F. Bach, and A. Rudi, “Non-parametric Models for Non-negative Functions” url: https://arxiv.org/abs/2007.03926
"""
function integral(a::PSDModel{T}, p::Function, χ::Domain; 
                    quadrature_method=gausslegendre,
                    amount_quadrature_points=20) where {T<:Number}
    M_p = zeros(size(a.B))

    x, w = quadrature_method(amount_quadrature_points)

    l = leftendpoint(χ)
    r = rightendpoint(χ)
    x .*= ((r - l)/2)
    x .+= ((r + l)/2)

    @inline to_int(x, i, j) = a.k(x, a.X[i]) * a.k(x, a.X[j]) * p(x)
    for i in CartesianIndices(a.B)
        M_p[i] = ((r - l)/2) * dot(w, to_int.(x, i[1], i[2]))
    end

    # tr(A * W_P) = tr(V B V^T * V^-T W_P V^-1) = tr(V B M_p V^-1)
    # = tr(B M_p V^-1 V) = tr(B M_p)
    return tr(a.B * M_p)
end

function Base.:+(a::PSDModel, 
                b::PSDModel)
    @error "Not implemented"
    return nothing
end

function Base.:-(
    a::PSDModel,
    b::PSDModel
)
    @error "Not implemented"
    return nothing
end

Base.:*(a::PSDModel, b::Number) = b * a
function Base.:*(a::Number, b::PSDModel)
    return PSDModel(
        a * b.B,
        b.k,
        b.X
    )
end

function Base.:*(
    a::PSDModel,
    b::PSDModel
)
    @error "Not implemented"
    return nothing
end

end # module PositiveSemidefiniteModels
