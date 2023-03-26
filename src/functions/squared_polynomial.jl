

struct SquaredPolynomialMatrix{d, T}
    Φ::FMTensorPolynomial{d, T}
    int_dim::Vector{Int}
    int_Fun::Vector{AbstractMatrix{Fun}}
    function SquaredPolynomialMatrix(Φ::FMTensorPolynomial{d, T}, int_dim::Vector{Int}) where {d, T}
        int_Fun = [Matrix{Fun}(undef, Φ.highest_order+1, Φ.highest_order+1) for _=1:length(int_dim)]
        
        for i=1:Φ.highest_order+1
            for j=i:Φ.highest_order+1
                for k in int_dim
                    f1 = Fun(Φ.space.spaces[k], [zeros(T, k);Φ.normal_factor[k][i]])
                    f2 = Fun(Φ.space.spaces[k], [zeros(T, k);Φ.normal_factor[k][j]])
                    res = Integral() * (f1 * f2)
                    int_Fun[k][i, j] = res
                    int_Fun[k][j, i] = res
                end
            end
        end
        return new{d, T}(Φ, int_dim, int_Fun)
    end
end

function (a::SquaredPolynomialMatrix)(x::PSDdata{T}) where {T<:Number}
    vec = _eval(a.Φ, x, a.int_dim)
    M = vec * vec'
    for i = 1:size(M, 1)
        for j = i:size(M, 2)
            for k in a.int_dim
                i1 = σ_inv(a.Φ, i)[k]
                i2 = σ_inv(a.Φ, j)[k]
                res = a.int_Fun[k][i1, i2](x[k])
                M[i, j] *= res
            end
        end
    end
    return Symmetric(M)
end