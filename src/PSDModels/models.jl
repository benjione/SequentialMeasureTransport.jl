include("kernel/PSDModelKernel.jl")
include("feature_map/PSDModelFM.jl")

# optimization algorithms acting on PSDModels
include("optimization.jl")

function PSDModel(k::Kernel, X::PSDDataVector{T}; kwargs...) where {T<:Number}
    B = diagm(ones(Float64, length(X)))
    return PSDModelKernel(Hermitian(B), k, X; 
                    _filter_kwargs(kwargs, _PSDModelKernel_kwargs)...)
end

PSDModel(Φ::Function, N::Int; kwargs...) = PSDModel{Float64}(Φ, N; kwargs...)
function PSDModel{T}(Φ::Function, N::Int; sparse=false, kwargs...) where {T<:Number}
    B = if sparse
        spdiagm(ones(Float64, N))
    else
        diagm(ones(Float64, N))
    end
    return PSDModelFM{T}(Hermitian(B), Φ; 
                    _filter_kwargs(kwargs, _PSDModelFM_kwargs)...)
end

PSDModel(sp::Space, ten_size::Int; kwargs...) = PSDModel{Float64}(sp, ten_size; kwargs...)
function PSDModel{T}(sp::Space, ten_size::Int; kwargs...) where {T<:Number}
    return PSDModel{T}(sp, :trivial, ten_size; kwargs...)
end
PSDModel(sp::Space, tensorizer::Symbol, ten_size::Int; kwargs...) = PSDModel{Float64}(sp, tensorizer, ten_size; kwargs...)
function PSDModel{T}(sp::Space, tensorizer::Symbol, ten_size::Int; 
                     sparse=false, mapping=nothing, kwargs...) where {T<:Number}
    Φ, N = if tensorizer == :trivial
        trivial_TensorPolynomial(T, sp, ten_size), ten_size
    elseif tensorizer == :downward_closed
        poly = downwardClosed_Polynomial(T, sp, ten_size)
        poly, max_N(poly.ten)
    else
        throw(error("Tensorizer not implemented"))
    end
    B = if sparse
        spdiagm(ones(T, N))
    else
        diagm(ones(T, N))
    end
    map = if mapping === nothing
        nothing
    elseif mapping == :algebraicOMF
            algebraicOMF{T}()
    elseif mapping == :logarithmicOMF
            logarithmicOMF{T}()
    else
        throw(error("Mapping not implemented"))
    end
    return PSDModelPolynomial(Hermitian(B), Φ, map; 
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

function PSDModel_from_polynomial(p::Fun)
    return @error("not implemented yet")
    return PSDModel{Float64}(p.sp, p.ten_size, sparse=true)
end

function (a::PSDModel{T})(x::PSDdata{T}) where {T<:Number}
    v = Φ(a, x)
    return dot(v, a.B, v)::T
end

# define this for Zygote, ForwardDiff, etc.
function (a::PSDModel)(x::PSDdata{T}) where {T<:Number}
    v = Φ(a, x)
    return dot(v, a.B, v)::T
end

function (a::PSDModel{T})(x::PSDdata{T}, B::AbstractMatrix{T}) where {T<:Number}
    v = Φ(a, x)
    return dot(v, B, v)::T
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
    x, w = quadrature_method(amount_quadrature_points)

    l = leftendpoint(χ)
    r = rightendpoint(χ)
    x .*= ((r - l)/2)
    x .+= ((r + l)/2)


    @inline Quad_point(w, x) = p(x) * w * (Φ(a,x) * Φ(a,x)')
    M_p = ((r - l)/2) * mapreduce(i->Quad_point(w[i], x[i]), .+, 1:length(x))

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
