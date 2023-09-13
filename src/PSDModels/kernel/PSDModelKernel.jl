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
            deepcopy(X)       # protect from further changes
        end
        new{T}(B, k, X)
    end
end

@inline _of_same_PSD(a::PSDModelKernel{T}, B::AbstractMatrix{T}) where {T<:Number} =
                                PSDModelKernel(Hermitian(B), a.k, a.X)

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
        optimization_method = :SDP,
        λ_1=1e-8,
        trace=false,
        B0=nothing,
        kwargs...
    ) where {T<:Number}
    K = kernelmatrix(k, X)

    N = length(X)

    f_A(i, A) = begin
        v = K[i,:]
        return dot(v, A, v)
    end
    f_A(A) = (1.0/N) * mapreduce(i-> (f_A(i, A) - Y[i])^2, +, 1:N) + λ_1 * tr(A)

    A0 = if B0===nothing
        ones(N,N)
    else
        B0
    end
    prob = create_SoS_opt_problem(optimization_method, 
                A0, f_A;
                trace=trace,
                _filter_kwargs(kwargs, 
                        _optimize_PSD_kwargs,
                        (:convex, :trace)
                )...
            )
    solution = optimize(prob)
    return PSDModelKernel(solution, k, X; 
                  _filter_kwargs(kwargs, _PSDModelKernel_kwargs)...)
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

