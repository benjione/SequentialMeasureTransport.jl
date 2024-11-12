

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
    _clenshaw(p, x) = begin
        l = ApproxFun.leftendpoint(p.space.domain)
        r = ApproxFun.rightendpoint(p.space.domain)
        ApproxFun.clenshaw(p.space, p.coefficients, (x-l)/(r-l) * 2.0 - 1.0)
    end
    
    vec = _eval(a.Φ, x, a.int_dim)::Vector{T}
    M = (vec * vec')::Matrix{T}
    eval_Fun = Vector{Symmetric{T, Matrix{T}}}(undef, length(a.int_dim))
    ind_j_list = NTuple{d, Int}[σ_inv(a.Φ, j) for j=1:size(M, 2)]
    
    for k_index=1:length(a.int_dim)
        @inbounds k = a.int_dim[k_index]
        tmp_mat = Matrix{T}(undef, size(a.int_Fun[k_index]))
        for i=1:size(a.int_Fun[k_index], 1)
            for j=i:size(a.int_Fun[k_index], 2)
                @inbounds tmp_mat[i, j] = _clenshaw(a.int_Fun[k_index][i, j], x[k])
            end
        end
        # no need to save as symmetric, since read in this order as well.
        @inbounds eval_Fun[k_index] = Symmetric(tmp_mat)
    end

    for i = 1:size(M, 1)
        @inbounds ind_i = ind_j_list[i]
        for j = i:size(M, 2)
            @inbounds ind_j = ind_j_list[j]
            mult_fac = one(T)
            for k_index=1:length(a.int_dim)
                @inbounds k = a.int_dim[k_index]
                @inbounds i1 = ind_i[k]
                @inbounds i2 = ind_j[k]
                @inbounds mult_fac *= eval_Fun[k_index][i1, i2]
            end
            @inbounds M[i, j] *= mult_fac
        end
    end
    return Symmetric(M)
end


function (a::SquaredPolynomialMatrix{d, T, S, FunType})(x::PSDdata{T2}) where {d, T<:Number, T2<:Number, S, FunType}
    _clenshaw(p, x) = begin
        l = ApproxFun.leftendpoint(p.space.domain)
        r = ApproxFun.rightendpoint(p.space.domain)
        ApproxFun.clenshaw(p.space, p.coefficients, (x-l)/(r-l) * 2.0 - 1.0)
    end

    vec = _eval(a.Φ, x, a.int_dim)
    M = T2.(vec * vec')
    eval_Fun = Vector{Symmetric{T2, Matrix{T2}}}(undef, length(a.int_dim))
    for k_index=1:length(a.int_dim)
        k = a.int_dim[k_index]
        tmp_mat = zeros(T2, size(a.int_Fun[k_index]))
        for i=1:size(a.int_Fun[k_index], 1), j=i:size(a.int_Fun[k_index], 2)
            # tmp_mat[i, j] = a.int_Fun[k_index][i, j](x[k])
            tmp_mat[i, j] = _clenshaw(a.int_Fun[k_index][i, j], x[k])
        end
        # no need to save as symmetric, since read in this order as well.
        eval_Fun[k_index] = Symmetric(tmp_mat)
    end
    # eval_Fun = map((M, xk)->map(f->f(xk), M), a.int_Fun, x[a.int_dim])

    ind_j_list = NTuple{d, Int}[σ_inv(a.Φ, j) for j=1:size(M, 2)]
    for i = 1:size(M, 1)
        ind_i = σ_inv(a.Φ, i)
        for j = i:size(M, 2)
            # ind_j = σ_inv(a.Φ, j)
            ind_j = ind_j_list[j]
            @inbounds for k_index=1:length(a.int_dim)
                k = a.int_dim[k_index]
                i1 = ind_i[k]::Int
                i2 = ind_j[k]::Int
                M[i, j] *= eval_Fun[k_index][i1, i2]
            end
        end
    end
    return Symmetric(M)
end

