
"""
Stopping rule estimates condition so that
P(|X - μ| > δ) < p
where X is the sample mean and μ is the true mean
"""
_chebyshev_stopping_rule(δ, p, m, variance) = (1 - p) - (variance / (m * δ^2))

## following algorithm 2 from Bicher et al 2022
## adds samples to X, Y until stopping rule is satisfied
function sample_adaptive!(g, loss, X, Y, rand_gen, stopping_rule, δ, p; 
                N0=200, Nmax=100000, addmax=500, addmin=10, addbatch=50,
                broadcasted_g=false)
    mean_X = zero(Float64)
    mean_X2 = zero(Float64)
    i = 0
    for (x, y) in zip(X, Y)
        i += 1
        mean_X = ((i - 1)/i) * mean_X + (1/i) * loss(x, y)
        mean_X2 = ((i - 1)/i) * mean_X2 + (1/i) * loss(x, y)^2
    end
    ## check stopping rule
    i_before_add = i
    s_sq = (i/(i-1)) * (mean_X2 - mean_X^2)
    while (stopping_rule(δ, p, i, s_sq) < 0 && i < Nmax && (i - i_before_add) < addmax) || i < N0 || (i - i_before_add) < addmin
        i += 1
        for _ = 1:addbatch
            x = rand_gen()
            push!(X, x)
        end
        _Y = if broadcasted_g
            g(X[i:end])
        else
            [g(x) for x in X[i:end]]
        end
        for y in _Y
            push!(Y, y)
        end
        while length(Y) > i
            mean_X = ((i - 1)/i) * mean_X + (1/i) * loss(X[i], Y[i])
            mean_X2 = ((i - 1)/i) * mean_X2 + (1/i) * loss(X[i], Y[i])^2
            i += 1
        end
        s_sq = (i/(i-1)) * (mean_X2 - mean_X^2)
    end
    # println("stopping rule is $(stopping_rule(δ, p, i, s_sq)) at i = $i and estimated variance is $s_sq")
    # println("mean x is $mean_X and squared mean is $mean_X2")
    return X, Y
end