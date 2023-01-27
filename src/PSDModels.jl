module PSDModels

using LinearAlgebra, SparseArrays
using KernelFunctions: Kernel
using ProximalOperators: IndPSD, prox, prox!
import ProximalAlgorithms
import Base

export PSDModel

struct PSDModel{T<:Number}
    B::Hermitian{T, Matrix{T}}  # A is the PSD so that f(x) = ∑_ij k(x, x_i) * A * k(x, x_j)
    k::Kernel                   # k(x, y) is the kernel function
    X::Vector{T}                # X is the set of points for the feature map
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
    K = T[k(x, y) for x in X, y in X]

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
    return PSDModel{T}(solution, k, X)
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
    K = T[k(x, y) for x in X, y in X]
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

    return PSDModel{T}(B, k, X)
end


function (a::PSDModel)(x::T) where {T<:Number}
    v = a.k.(Ref(x), a.X)
    return v' * a.B * v
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
