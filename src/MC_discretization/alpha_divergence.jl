
function ϕ_a(α::T, u::T, v::T) where {T<:Number}
    t = u/v
    res = zero(T)
    if α == 1
        res = t * log(t) - t + 1
    elseif α == 0
        res = - log(t) + t - 1
    else
        res = (t^α - 1)/(α * (α - 1)) - (t - 1)/(α - 1)
    end
    return res
end

"""
    sample_for_α_div(α::T, d::Int, X::PSDDataVector{T}, Y::AbstractVector{T}, u, g, δ::T, prb_bound::T)

This function performs sampling for α-divergence computation for D_α(u||g). 
It samples until the variance of the α-divergence estimate is less than `δ` with probability `prb_bound` for the given `u` and `g`.

**Arguments**
- `α::T`: The value of α for the α-divergence.
- `d::Int`: The dimension of the random variable.
- `X::PSDDataVector{T}`: The vector of samples from the source distribution.
- `Y::AbstractVector{T}`: The vector of samples from the target distribution.
- `u`: A function that maps a random variable `x` to a sample from the target distribution `y = u(x)`.
- `g`: A function that maps a random variable `x` to the weight `g(x)` used in the α-divergence computation.
- `δ::T`: The tolerance for the variance of the α-divergence estimate.
- `prb_bound::T`: The bound on the probability of the α-divergence estimate being outside the tolerance.

**Returns**
- `X`: The updated vector of samples for Monte Carlo estimation.
- `Y`: The updated vector of samples from the target distribution.
"""
function sample_for_α_div(α::T, d::Int, 
                X::PSDDataVector{T}, 
                Y::AbstractVector{T}, 
                u, 
                g, 
                δ::T, 
                prb_bound::T
            ) where {T<:Number}
    if length(X) != length(Y)
        throw(ArgumentError("X and Y must have the same length."))
    end
    n = 0
    mean_vec, sq_mean_vec = T[], T[]
    mean = zero(T)
    sq_mean = zero(T)

    # first iterate through X and Y and check if variance bound for given g is satisfied
    while n < length(X)
        n += 1
        x = X[n]
        y = Y[n]
        g_x = g(x)
        weight = g_x
        Da_x = ϕ_a(α, y, g_x) * weight
        mean = (mean * (n - 1)/n) + (Da_x/n)
        sq_mean = (sq_mean * (n - 1)/n) + (mean^2/n)
        push!(mean_vec, mean)
        push!(sq_mean_vec, sq_mean)

        ## check if the variance is small enough
        if n > 1
            variance = sq_mean - mean^2
            if variance * prb_bound < δ
                break
            end
        end
    end
    # if variance is not small enough, sample until it is
    while true
        # sample
        n += 1
        x = rand(d)
        y = u(x)
        push!(X, x)
        push!(Y, y)
        
        g_x = g(x)
        weight = g_x
        Da_x = ϕ_a(α, y, g_x) * weight
        mean = (mean * (n - 1) + Da_x)/n
        sq_mean = (sq_mean * (n - 1) + mean^2)/n
        push!(mean_vec, mean)
        push!(sq_mean_vec, sq_mean)

        ## check if the variance is small enough
        if n > 1
            variance = sq_mean - mean^2
            if variance * prb_bound < δ
                break
            end
        end
    end
    return X, Y
end