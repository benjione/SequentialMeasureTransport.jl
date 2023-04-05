

struct SquaredPolynomialMatrix{d, T, S}
    Φ::FMTensorPolynomial{d, T, S}
    int_dim::Vector{Int}                    # dimension which are integrated over
    int_Fun::Vector{<:AbstractMatrix{Fun}}    # integral of Φ[i] * Φ[j] over dimension int_dim
    function SquaredPolynomialMatrix(Φ::FMTensorPolynomial{d, T, S}, int_dim::Vector{Int}) where {d, T, S}
        int_Fun = [Matrix{Fun}(undef, Φ.highest_order+1, Φ.highest_order+1) for _=1:length(int_dim)]
        
        for i=1:Φ.highest_order+1
            for j=i:Φ.highest_order+1
                for k_index=1:length(int_dim)
                    k = int_dim[k_index]
                    f1 = Fun(Φ.space.spaces[k], [zeros(T, i-1);Φ.normal_factor[k][i]])
                    f2 = Fun(Φ.space.spaces[k], [zeros(T, j-1);Φ.normal_factor[k][j]])
                    res = Integral() * (f1 * f2)
                    res = res - res(0.0)  ## let integral start from 0
                    int_Fun[k_index][i, j] = res
                    int_Fun[k_index][j, i] = res
                end
            end
        end
        return new{d, T, S}(Φ, int_dim, int_Fun)
    end
end

function (a::SquaredPolynomialMatrix{<:Any, T})(x::PSDdata{T}) where {T<:Number}
    vec = _eval(a.Φ, x, a.int_dim)::Vector{T}
    M = (vec * vec')::Matrix{T}
    eval_Fun = Symmetric{T, Matrix{T}}[]
    for k_index=1:length(a.int_dim)
        k = a.int_dim[k_index]
        tmp_mat = zeros(T, size(a.int_Fun[k_index]))
        for i=1:size(a.int_Fun[k_index], 1), j=i:size(a.int_Fun[k_index], 2)
            tmp_mat[i, j] = a.int_Fun[k_index][i, j](x[k])
        end
        sym_mat = Symmetric(tmp_mat)
        push!(eval_Fun, sym_mat)
    end
    # eval_Fun = map((M, xk)->map(f->f(xk), M), a.int_Fun, x[a.int_dim])

    for i::Int = 1:size(M, 1)
        ind_i = σ_inv(a.Φ, i)
        for j::Int = i::Int:size(M, 2)
            ind_j = σ_inv(a.Φ, j)
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