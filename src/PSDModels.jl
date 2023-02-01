module PSDModels

using LinearAlgebra, SparseArrays
using KernelFunctions: Kernel, kernelmatrix
using ProximalOperators: IndPSD, prox, prox!
using DomainSets
using FastGaussQuadrature: gausslegendre
import ForwardDiff as FD
import ProximalAlgorithms
import Base

export PSDModel
export gradient, fit!, integral

struct PSDModel{T<:Number}
    B::Hermitian{T, Matrix{T}}  # B is the PSD so that f(x) = ∑_ij k(x, x_i) * B * k(x, x_j)
    k::Kernel                   # k(x, y) is the kernel function
    X::Vector{T}                # X is the set of points for the feature map
    function PSDModel(B::Hermitian{T, Matrix{T}}, 
                        k::Kernel, 
                        X::Vector{T}
                    ) where {T<:Number}
        new{T}(B, k, X)
    end
end

function PSDModel(k::Kernel, X::Vector{T}) where {T<:Number}
    B = ones(length(X), length(X))
    return PSDModel(Hermitian(B), k, X)
end

function PSDModel(
                X::Vector{T}, 
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

function PSDModel_gradient_descent(
                        X::Vector{T},
                        Y::Vector{T},
                        k::Kernel;
                        λ_1=1e-8,
                        trace=false,
                        maxit=5000,
                        tol=1e-6,
                        B0=nothing,
                    ) where {T<:Number}
    K = kernelmatrix(k, X)

    N = length(X)
    
    f_A(i, A::AbstractMatrix) = begin
        v = K[i,:]
        return v' * A * v
    end
    f_A(A::AbstractMatrix) = (1.0/N) * mapreduce(i-> (f_A(i, A) - Y[i])^2, +, 1:N) + λ_1 * tr(A)

    psd_constraint = IndPSD()

    verbose_solver = trace ? true : false

    A0 = if B0===nothing
        ones(N,N)
    else
        B0
    end
    solver = ProximalAlgorithms.FastForwardBackward(maxit=maxit, tol=tol, verbose=verbose_solver)
    solution, _ = solver(x0=A0, f=f_A, g=psd_constraint)

    solution = Hermitian(solution)
    return PSDModel(solution, k, X)
end

function PSDModel_direct(
                X::Vector{T}, 
                Y::Vector{T}, 
                k::Kernel;
                regularize_kernel=true,
                cond_thresh=1e10,
                λ_1=1e-8,
                trace=false,
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

    return PSDModel(B, k, X)
end

fit!(a::PSDModel, 
        X::Vector{T}, 
        Y::Vector{T}; 
        kwargs...
    ) where {T<:Number} = fit!(a, X, Y, ones(T, length(X)); kwargs...)
function fit!(a::PSDModel{T}, 
                X::Vector{T}, 
                Y::Vector{T},
                weights::Vector{T}; 
                λ_1=1e-8,
                trace=false,
                maxit=5000,
                tol=1e-6
            ) where {T<:Number}
    N = length(X)


    f_A(A::AbstractMatrix) = begin
        (1.0/N) * mapreduce(i-> weights[i]*(a(X[i], A) - Y[i])^2, +, 1:N) + λ_1 * tr(A)
    end

    psd_constraint = IndPSD()

    verbose_solver = trace ? true : false

    solver = ProximalAlgorithms.FastForwardBackward(maxit=maxit, tol=tol, verbose=verbose_solver)
    solution, _ = solver(x0=Matrix(a.B), f=f_A, g=psd_constraint)

    solution = Hermitian(solution)
    set_coefficients!(a, solution)
end

function (a::PSDModel)(x::T) where {T<:Number}
    v = a.k.(Ref(x), a.X)
    return v' * a.B * v
end

function (a::PSDModel)(x::T, B::AbstractMatrix{T}) where {T<:Number}
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
