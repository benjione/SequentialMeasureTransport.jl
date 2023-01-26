module PositiveSemidefiniteModels

using LinearAlgebra, SparseArrays
using KernelFunctions: Kernel
using ProximalOperators: IndPSD, prox, prox!
import Base

export PSDModel

struct PSDModel{T<:Number}
    B::Hermitian{T, Matrix{T}}  # A is the PSD so that f(x) = ∑_ij k(x, x_i) * A * k(x, x_j)
    k::Kernel                   # k(x, y) is the kernel function
    X::Vector{T}                # X is the set of points for the feature map
end

function PositiveSemidefiniteModel(
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
    return PositiveSemidefiniteModel(
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
