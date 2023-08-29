

struct SquaredPolynomialMatrix{d, T, S, FunType}
    Φ::FMTensorPolynomial{d, T, S}
    int_dim::Vector{Int}                    # dimension which are integrated over
    int_Fun::Vector{FunType}    # integral of Φ[i] * Φ[j] over dimension int_dim
    function SquaredPolynomialMatrix(Φ::FMTensorPolynomial{d, T, S}, int_dim::Vector{Int}; C=nothing) where {d, T, S}
        if C === nothing
            C = leftendpoint(Φ.space.spaces[int_dim].domain)
        end
        
        M_type = typeof(Fun(Φ.space.spaces[int_dim[1]], rand(2)))
        int_Fun = Matrix{M_type}[Matrix{M_type}(undef, Φ.highest_order+1, Φ.highest_order+1) for _=1:length(int_dim)]
        
        for i=1:Φ.highest_order+1
            for j=i:Φ.highest_order+1
                for k_index=1:length(int_dim)
                    k = int_dim[k_index]
                    f1 = Fun(Φ.space.spaces[k], T[zeros(T, i-1);Φ.normal_factor[k][i]])
                    f2 = Fun(Φ.space.spaces[k], T[zeros(T, j-1);Φ.normal_factor[k][j]])
                    res = Integral() * (f1 * f2)
                    res = res - res(C)  ## let integral start from 0
                    int_Fun[k_index][i, j] = res
                    int_Fun[k_index][j, i] = res
                end
            end
        end
        return new{d, T, S, typeof(int_Fun[1])}(Φ, int_dim, int_Fun)
    end
end

function (a::SquaredPolynomialMatrix{d, T, S, FunType})(x::PSDdata{T}) where {d, T<:Number, S, FunType}
    vec = _eval(a.Φ, x, a.int_dim)::Vector{T}
    M = (vec * vec')::Matrix{T}
    eval_Fun = Vector{Symmetric{T, Matrix{T}}}(undef, length(a.int_dim))
    for k_index=1:length(a.int_dim)
        k = a.int_dim[k_index]
        tmp_mat = zeros(T, size(a.int_Fun[k_index]))
        for i=1:size(a.int_Fun[k_index], 1), j=i:size(a.int_Fun[k_index], 2)
            tmp_mat[i, j] = a.int_Fun[k_index][i, j](x[k])
        end
        # no need to save as symmetric, since read in this order as well.
        eval_Fun[k_index] = Symmetric(tmp_mat)
    end
    # eval_Fun = map((M, xk)->map(f->f(xk), M), a.int_Fun, x[a.int_dim])

    ind_j_list = Vector{Int}[σ_inv(a.Φ, j) for j=1:size(M, 2)]
    for i::Int = 1:size(M, 1)
        ind_i = σ_inv(a.Φ, i)
        for j::Int = i::Int:size(M, 2)
            # ind_j = σ_inv(a.Φ, j)
            ind_j = ind_j_list[j]
            @inbounds for k_index=1:length(a.int_dim)
                k = a.int_dim[k_index]
                i1 = ind_i[k]::Int
                i2 = ind_j[k]::Int
                M[i, j] *= eval_Fun[k_index][i1, i2]::T
            end
        end
    end
    return Symmetric(M)
end

