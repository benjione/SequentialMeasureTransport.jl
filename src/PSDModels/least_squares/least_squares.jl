


function _create_vandermonde_matrix(PSD_model::PSDModel{T}, x::PSDDataVector{T}) where {T}
    m = length(x)
    n = size(PSD_model.B)[1]
    n = (n*(n+1))÷2
    M = zeros(m, n)
    for i = 1:m
        v = Φ(PSD_model, x[i])
        M_i = v*v'
        M_i = M_i + M_i' - Diagonal(diag(M_i))
        M_i = Hermitian_to_low_vec(M_i)
        M[i, :] = M_i
    end
    return M
end


least_squares_fit!(model, X::PSDDataVector{T}, Y; kwargs...) where {T <: Number} = 
                        weighted_least_squares_fit!(model, X, Y, ones(T, length(Y)); kwargs...)

function weighted_least_squares_fit!(
            model::PSDModel{T},
            X::PSDDataVector{T},
            Y::AbstractVector{T},
            W::AbstractVector{T};
            kwargs...
        ) where {T<:Number}
    M = _create_vandermonde_matrix(model, X)
    M2 = M' * (W .* M)
    Y2 = M' * (W .* Y)
    B_vec = M2 \ Y2
    B = low_vec_to_Symmetric(B_vec)
    # return B
    # remove negative eigenvalues
    # S, U = eigen(B)
    # S[S .< 0.0] .= 1.0
    # B_ret = U * Diagonal(S) * inv(U)
    B_ret = nothing
    if typeof(model) <: PSDModelPolynomial
        D, C = get_semialgebraic_domain_constraints(model)
        B_ret = _closest_PSD_JuMP!(Hermitian(B), mat_list=D, coef_list=C; kwargs...)
        # B_ret = _closest_PSD_JuMP!(Hermitian(B))
        # return B_ret
    else
        B_ret = _closest_PSD_JuMP!(Hermitian(B); kwargs...)
    end

    set_coefficients!(model, Hermitian(B_ret))
    return nothing
end


function iteratively_reweighted_least_squares!(
            model::PSDModel{T},
            X::PSDDataVector{T},
            Y::AbstractVector{T},
            weight::Function;
            max_iter = 1000,
            max_change = 1e-6,
        ) where {T<:Number}
    W = ones(T, length(Y))
    change = Inf
    k = 1
    while (k < max_iter) && (change > max_change)
        weighted_least_squares_fit!(model, X, Y, W)
        W_new = weight.(model.(X), Y)
        change = norm(W - W_new, Inf)
        W = W_new
        k += 1
    end
    return nothing
end