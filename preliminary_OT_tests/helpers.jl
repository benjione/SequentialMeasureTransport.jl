using LinearAlgebra
using FastGaussQuadrature

function compute_Sinkhorn(rng, left_marg_d, right_marg_d, c, ϵ; iter=200)
    M_epsilon = [exp(-c([x,y])/ϵ) for x in rng, y in rng]
    left_mat = [left_marg_d([x]) for x in rng]
    right_mat = [right_marg_d([x]) for x in rng]
    left_margin(M) = M * ones(size(M, 1))
    right_margin(M) = M' * ones(size(M, 2))
    for _=1:iter
        M_epsilon = diagm(left_mat ./ left_margin(M_epsilon)) * M_epsilon
        M_epsilon = M_epsilon * diagm(right_mat ./ right_margin(M_epsilon))
    end
    M_epsilon = length(rng)^2 * M_epsilon / sum(M_epsilon)
    # for (i, ii) in enumerate(rng), (j, jj) in enumerate(rng)
    #     M_epsilon[i, j] *= 1/(left_marg_d([ii]) * right_marg_d([jj]))
    # end
    return M_epsilon
end


function left_pdf(sampler, x)
    p, w = FastGaussQuadrature.gausslegendre(50)
    ## scale to [0, 1]
    p .= (p .+ 1) ./ 2
    w .= w ./ 2
    return sum(w .* [pdf(sampler, [x, y]) for y in p])
end

function right_pdf(sampler, x)
    p, w = FastGaussQuadrature.gausslegendre(50)
    ## scale to [0, 1]
    p .= (p .+ 1) ./ 2
    w .= w ./ 2
    return sum(w .* [pdf(sampler, [y, x]) for y in p])
end

function compute_Sinkhorn_distance(c, 
            smp::SMT.ConditionalMapping{d}; 
            N=10000) where {d}
    _d = d ÷ 2
    X = rand(smp, N)
    return mapreduce(x -> c(x[1:_d], x[_d+1:end]), +, X) / N
end

function compute_Sinkhorn_distance(c, 
            M_sink::AbstractArray{T, d},
            rng) where {d, T<:Number}
    _d = d ÷ 2
    density = (rng[end] - rng[1]) / length(rng)
    res = zero(T)
    for ((i, x), (j, y)) in Iterators.product(enumerate(rng), enumerate(rng))
        res += c(x, y) * M_sink[i, j] * density^d
    end
    return res
end