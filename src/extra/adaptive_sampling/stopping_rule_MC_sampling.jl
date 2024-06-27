abstract type SamplingStruct{T, d} end


struct FixedSampleStruct{T<:Number, d} <: SamplingStruct{T, d}
    Nmin::Int
    Nmax::Int
    add_per_iter::Int
    rand_gen
    function FixedSampleStruct{T, d}(Nmin, Nmax, add_per_iter) where {T<:Number, d}
        rand_gen = ()->rand(T, d)
        new{T, d}(Nmin, Nmax, add_per_iter, rand_gen)
    end
end


function sample!(fixed_s::FixedSampleStruct{T}, π_tar, loss, 
                X::PSDDataVector{T}, Y::AbstractVector{T};
                broadcasted_target=false) where {T<:Number}
    i = length(X)
    if i < fixed_s.Nmin
        for _ = i:fixed_s.Nmin
            x = fixed_s.rand_gen()
            push!(X, x)
        end
        _Y = if broadcasted_target
            π_tar(X[i:end])
        else
            [π_tar(x) for x in X[i:end]]
        end
        for y in _Y
            push!(Y, y)
        end
        i = fixed_s.Nmin
        return
    end
    j = i
    while j < fixed_s.Nmax && j < i + fixed_s.add_per_iter
        x = fixed_s.rand_gen()
        push!(X, x)
    end
    _Y = if broadcasted_target
        π_tar(X[i+1:end])
    else
        [π_tar(x) for x in X[i+1:end]]
    end
    for y in _Y
        push!(Y, y)
    end
    return
end


struct AdaptiveSamplingStruct{T<:Number, d} <: SamplingStruct{T, d}
    δ::T
    p::T
    Nmin::Int
    Nmax::Int
    stopping_rule
    rand_gen
    addmax::Int
    addbatch::Int
    function AdaptiveSamplingStruct{T, d}(δ, p, Nmin, Nmax, stopping_rule, rand_gen; 
                        addmax=1000, addbatch=50) where {T<:Number, d}
        new{T, d}(δ, p, Nmin, Nmax, stopping_rule, rand_gen, addmax, addbatch)
    end
    function AdaptiveSamplingStruct{T, d}(δ, p; Nmin=200, Nmax=100000, 
                        stopping_rule=_chebyshev_stopping_rule, 
                        addmax=1000, addbatch=50) where {T<:Number, d}
        rand_gen = ()->rand(T, d)
        new{T, d}(δ, p, Nmin, Nmax, stopping_rule, rand_gen, addmax, addbatch)
    end
end

"""
Stopping rule estimates condition so that
P(|X - μ| > δ) < p
where X is the sample mean and μ is the true mean
"""
_chebyshev_stopping_rule(δ, p, m, variance) = (1 - p) - (variance / (m * δ^2))

## following algorithm 2 from Bicher et al 2022
## adds samples to X, Y until stopping rule is satisfied
function sample!(adap_s::AdaptiveSamplingStruct, π_tar, loss, X, Y;
                broadcasted_target=false, trace=false)
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
    while (adap_s.stopping_rule(adap_s.δ, adap_s.p, i, s_sq) < 0 && i < adap_s.Nmax && (i - i_before_add) < adap_s.addmax) || i < adap_s.Nmin
        i += 1
        for _ = 1:adap_s.addbatch
            x = adap_s.rand_gen()
            push!(X, x)
        end
        _Y = if broadcasted_target
            π_tar(X[i:end])
        else
            [π_tar(x) for x in X[i:end]]
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
    println("Stopping rule is $(adap_s.stopping_rule(adap_s.δ, adap_s.p, i, s_sq)) with $(i - i_before_add) samples added.")
    return X, Y
end