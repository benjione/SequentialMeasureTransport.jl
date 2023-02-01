module PSDModels

using LinearAlgebra, SparseArrays
using KernelFunctions: Kernel, kernelmatrix
using ProximalOperators: IndPSD, prox, prox!
import ForwardDiff as FD
import ProximalAlgorithms
import Base

export PSDModel
export gradient, fit!

struct PSDModel{T<:Number}
    B::Hermitian{T, Matrix{T}}  # A is the PSD so that f(x) = ∑_ij k(x, x_i) * A * k(x, x_j)
    k::Kernel                   # k(x, y) is the kernel function
    X::Vector{T}                # X is the set of points for the feature map
    function PSDModel(k::Kernel, X::Vector{T}) where {T<:Number}
        B = ones(length(X), length(X))
        return new{T}(Hermitian(B), k, X)
    end
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

    return PSDModel{T}(B, k, X)
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
