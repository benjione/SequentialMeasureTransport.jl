using LinearAlgebra
using FastGaussQuadrature
using ImageFiltering

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

function compute_Sinkhorn(rng, left_array::AbstractVector{T}, right_array::AbstractVector{T}, 
                            c, ϵ; iter=200, domain_size=1.0) where {T<:Number}
    M_epsilon = zeros(length(rng), length(rng))
    for (i, x) in enumerate(rng), (j, y) in enumerate(rng)
        M_epsilon[i, j] = exp(-c(SA[x...],SA[y...])/ϵ)
    end
    proj_mat = ones(size(M_epsilon, 1))
    left_margin(M) = M * proj_mat
    right_margin(M) = M' * proj_mat
    for _=1:iter
        M_epsilon .= diagm(left_array ./ left_margin(M_epsilon)) * M_epsilon
        M_epsilon .= M_epsilon * diagm(right_array ./ right_margin(M_epsilon))
    end
    M_epsilon = length(rng)^2 * M_epsilon / (sum(M_epsilon) * domain_size^2)
    # for (i, ii) in enumerate(rng), (j, jj) in enumerate(rng)
    #     M_epsilon[i, j] *= 1/(left_marg_d([ii]) * right_marg_d([jj]))
    # end
    return M_epsilon
end

function Barycentric_map_from_sinkhorn(M_sink, left_marg_vec, right_marg_vec, i)
    res = SA[0.0, 0.0, 0.0]
    for j in 1:length(right_marg_vec)
        res += M_sink[i, j] * right_marg_vec[j]
    end
    return (1/left_marg_vec[i]) * res
end

"""
Implemented for 1D distributions as in:
    Justin Solomon, Fernando de Goes, Gabriel Peyré, Marco Cuturi, 
    Adrian Butscher, Andy Nguyen, Tao Du, and Leonidas Guibas. 2015. 
    Convolutional wasserstein distances: efficient optimal transportation 
    on geometric domains. ACM Trans. Graph. 34, 4, Article 66 (August 2015), 
    11 pages. https://doi.org/10.1145/2766963
"""
function compute_Sinkhorn_Wasserstein_barycenter(rng, 
                                list_marg::Vector{Function},
                                weights::Vector{T},
                                ϵ; iter=200) where {T<:Number}
    v = ones(length(list_marg), length(rng))
    w = ones(length(list_marg), length(rng))
    d = ones(length(list_marg), length(rng))
    marg_vec = [list_marg[i](x) for i in 1:length(list_marg), x in rng]
    a = 1 / length(rng)^2
    mu = ones(length(rng))
    for _=1:iter
        mu .= 1.0
        for i=1:length(list_marg)
            w[i, :] .= marg_vec[i, :] ./ imfilter(a * v[i, :], Kernel.gaussian((ϵ,)))
            d[i, :] .= v[i, :] .* imfilter(a * w[i, :], Kernel.gaussian((ϵ,)))
            mu .= mu .* (w[i, :].^ weights[i]) 
        end

        for i=1:length(list_marg)
            v[i, :] .= (v[i, :] .* mu) ./ d[i, :]
        end
    end
    return mu
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