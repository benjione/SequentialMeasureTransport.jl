import SpecialFunctions: erf

struct KernelTraceModel{T} <: TraceModel{T}
    X_dim::AbstractVector{T}
    X_not_dim::PSDDataVector{T}
    B::Hermitian{T, <:AbstractMatrix{T}}
    k::Kernel
    dim::Int
    function KernelTraceModel(X::PSDDataVector{T}, 
        A::Hermitian{T, <:AbstractMatrix{T}}, 
        k::Kernel,
        dim::Int) where {T}
        X_dim = [y[dim] for y in X]
        X_not_dim = [y[1:dim-1] ∪ y[dim+1:end] for y in X]
        new{T}(X_dim, X_not_dim, A, k, dim)
    end
end

function ΦΦT(a::KernelTraceModel{T}, x::PSDdata{T}) where {T<:Number}
    X_dim = a.X_dim
    l = 1/a.k.transform.s[1]
    tmp = (X_dim .+ X_dim') ./ 2.0
    tmp = (x[a.dim] .- tmp) ./ l
    tmp = erf.(tmp) .+ 1.0
    # K = 2.0 * ((X_dim .+ X_dim') ./ 2.0).^2 - 2.0 *(X_dim * X_dim')
    K = 0.5*((X_dim .- X_dim')).^2
    tmp = (sqrt(π)/2) * l * tmp .* exp.(-K ./ (2.0*l^2))
    _d = length(x)
    if _d == 1
        return tmp
    end
    v = kernelmatrix(a.k, a.X_not_dim, x[1:a.dim-1] ∪ x[a.dim+1:end])
    return (v * v') .* tmp
end