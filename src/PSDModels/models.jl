


include("kernel/PSDModelKernel.jl")
include("feature_map/PSDModelFM.jl")

function PSDModel(k::Kernel, X::PSDDataVector{T}; kwargs...) where {T<:Number}
    B = diagm(ones(Float64, length(X)))
    return PSDModelKernel(Hermitian(B), k, X; 
                    _filter_kwargs(kwargs, _PSDModelKernel_kwargs)...)
end

PSDModel(Φ::Function, N::Int; kwargs...) = PSDModel{Float64}(Φ, N; kwargs...)
function PSDModel{T}(Φ::Function, N::Int; kwargs...) where {T<:Number}
    B = diagm(ones(Float64, N))
    return PSDModelFM{T}(Hermitian(B), Φ; 
                    _filter_kwargs(kwargs, _PSDModelFM_kwargs)...)
end

PSDModel(sp::Space, N::Int; kwargs...) = PSDModel{Float64}(sp, N; kwargs...)
function PSDModel{T}(sp::Space, N::Int; kwargs...) where {T<:Number}
    B = diagm(ones(Float64, N))
    return PSDModelFMPolynomial{T}(Hermitian(B), sp; 
                    _filter_kwargs(kwargs, _PSDModelFM_kwargs)...)
end

PSDModel(sp::Space, tensorizer::Symbol, N::Int; kwargs...) = PSDModel{Float64}(sp, tensorizer, N; kwargs...)
function PSDModel{T}(sp::Space, tensorizer::Symbol, N::Int; kwargs...) where {T<:Number}
    B = diagm(ones(Float64, N))

    Φ = if tensorizer == :trivial
        trivial_TensorPolynomial(sp, N)
    else
        @error "Tensorizer not implemented"
    end
    return PSDModelFMTensorPolynomial{T}(Hermitian(B), Φ; 
                    _filter_kwargs(kwargs, _PSDModelFM_kwargs)...)
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

function (a::PSDModel)(x::PSDdata{T}) where {T<:Number}
    v = Φ(a, x)
    return v' * a.B * v
end

function (a::PSDModel)(x::PSDdata{T}, B::AbstractMatrix{T}) where {T<:Number}
    v = Φ(a, x)
    return v' * B * v
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
        let K = reduce(hcat, Φ.(Ref(a), X))
            (i, A::AbstractMatrix) -> begin
                v = K[:,i]
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
        let K = reduce(hcat, Φ.(Ref(a), X))
            (i, A::AbstractMatrix) -> begin
                v = K[:, i]
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


function set_coefficients!(a::PSDModel{T}, B::Hermitian{T}) where {T<:Number}
    a.B .= B
end

function gradient(a::PSDModel{T}, x::T) where {T<:Number}
    # ∇v = FD.derivative((y)->a.k.(Ref(y), a.X), x)
    # v = a.k.(Ref(x), a.X)
    # return 2 * ∇v' * a.B * v
    
    # ForwardDiff faster than manual implementation
    return FD.derivative(a, x)
end

function parameter_gradient(a::PSDModel{T}, x::T) where {T<:Number}
    v = Φ(a, x)
    # ∇B = FD.derivative((B)->v' * B * v, a.B)

    ∇B = Matrix{T}(undef, size(a.B)...)
    @inbounds @simd for i in CartesianIndices(a.B)
        ∇B[i] = v[i[1]] * v[i[2]]
    end
    return ∇B
end

function integrate(a::PSDModel{T}, χ::Domain; kwargs...) where {T<:Number}
    return integrate(a, x->1.0, χ; kwargs...)
end


"""
integrate(a::PSDModel{T}, p::Function, χ::Domain; quadrature_method=gausslegendre, amount_quadrature_points=10) where {T<:Number}

returns ``\\int_χ p(x) a(x) dx``. The idea of the implementation is from proposition 4 in [1]. 
The integral is approximated by a quadrature rule. The default quadrature rule is Gauss-Legendre.

[1] U. Marteau-Ferey, F. Bach, and A. Rudi, “Non-parametric Models for Non-negative Functions” url: https://arxiv.org/abs/2007.03926
"""
function integrate(a::PSDModel{T}, p::Function, χ::Domain; 
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
    return _of_same_PSD(b, a * b.B)
end

"""
mul!(a::PSDModel, b::Number)

In place multiplication with a number.
"""
mul!(a::PSDModel, b::Number) = a.B .= b * a.B; nothing

function Base.:*(
    a::PSDModel,
    b::PSDModel
)
    @error "Not implemented"
    return nothing
end
