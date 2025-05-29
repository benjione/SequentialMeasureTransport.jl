

"""
Map to a gaussian reference distribution.
"""
struct SuperHeavyReference{d, dC, T} <: ReferenceMap{d, dC, T} 
    c::T
    D::T
    function SuperHeavyReference{d, dC, T}(c::T) where {d, dC, T<:Number}
        @assert 0 < c < 1
        D = 0.5 - 0.5 * erf(log(c)/sqrt(- 2 * log(c)))
        new{d, dC, T}(c, D)
    end
    function SuperHeavyReference{d,T}(c::T) where {d, T<:Number}
        SuperHeavyReference{d, 0, T}(c)
    end
    function SuperHeavyReference{d,T}() where {d, T<:Number}
        SuperHeavyReference{d, 0, T}(T(0.5))
    end
    function SuperHeavyReference{d,dC,T}() where {d, dC, T<:Number}
        new{d, dC, T}(T(0.5))
    end
end

function Distributions.pdf(Rmap::SuperHeavyReference{d, <:Any, T}, x::PSDdata{T}) where {d, T<:Number}
    @assert length(x) == d
    x2 = abs.(x) .+ Rmap.c
    σ = sqrt(-log(Rmap.c))
    return prod((1 ./ (Rmap.D * 2 * σ * x2 * sqrt(2π))) .* exp.(- log.(x2).^2 / (2*σ^2)))
end

function sample_reference(map::SuperHeavyReference{d, <:Any, T}) where {d, T<:Number}
    randn(T, d) * map.σ
    return error("Not implemented yet")
end

function sample_reference(map::SuperHeavyReference{d, <:Any, T}, n::Int) where {d, T<:Number}
    eachcol(randn(T, d, n) * map.σ)
    return error("Not implemented yet")
end

function SMT.pushforward(
        m::SuperHeavyReference{d, <:Any, T}, 
        x::PSDdata{T2}
    ) where {d, T<:Number, T2<:Number}
    @assert length(x) == d
    C = sqrt(-2*log(m.c))
    function CDF2(x)
        if x ≤ 0
            return 0.0
        end
        return 0.5*((1/(2*m.D)) * (erf(log(x+m.c)/C) - one(T)) + one(T))
    end
    function CDF1(x)
        if x ≥ 0
            return 0.5
        end
        return -CDF2(-x) + 0.5
    end
    return CDF1.(x) .+ CDF2.(x)

end


function SMT.pullback(
        m::SuperHeavyReference{d, <:Any, T}, 
        u::PSDdata{T2}
    ) where {d, T<:Number, T2<:Number}
    @assert length(u) == d
    z = erfcinv.((4*m.D*u))
    C = sqrt(-2*log(m.c))
    ap = x->ifelse(x < 0, m.c - exp(-x * C), exp(x*C) - m.c)
    return ap.(z)
end


function SMT.Jacobian(
        m::SuperHeavyReference{d, <:Any, T}, 
        x::PSDdata{T2}
    ) where {d, T<:Number, T2<:Number}
    @assert length(x) == d
    return Distributions.pdf(m, x)
end


function SMT.inverse_Jacobian(
        mapping::SuperHeavyReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    # inverse function theorem
    return 1/Jacobian(mapping, pullback(mapping, u))
end

function SMT.log_Jacobian(
        m::SuperHeavyReference{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    @assert length(x) == d
    mapreduce(xi->Distributions.logpdf(Distributions.Normal(0, m.σ), xi), +, x)
end

function SMT.inverse_log_Jacobian(
        mapping::SuperHeavyReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    # inverse function theorem
    return -SMT.log_Jacobian(mapping, SMT.pullback(mapping, u))
end

function SMT.marginal_pushforward(
        m::SuperHeavyReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    return 0.5 * (1 .+ erf.(x ./ (m.σ * sqrt(2))))
end

function SMT.marginal_pullback(
        m::SuperHeavyReference{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(u) == d-dC
    return sqrt(2) * m.σ * erfcinv.(2.0 .- 2*u)
end

function SMT.marginal_Jacobian(
        m::SuperHeavyReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    mapreduce(xi->Distributions.pdf(Distributions.Normal(0, m.σ), xi), *, x)
end

function SMT.marginal_inverse_Jacobian(
        mapping::SuperHeavyReference{<:Any, <:Any, T}, 
        u::PSDdata{T}
    ) where {T<:Number}
    # inverse function theorem
    return 1/SMT.marginal_Jacobian(mapping, SMT.marginal_pullback(mapping, u))
end

function SMT.marginal_log_Jacobian(
        m::SuperHeavyReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    @assert length(x) == d-dC
    mapreduce(xi->Distributions.logpdf(Distributions.Normal(0, m.σ), xi), +, x)
end

function SMT.marginal_inverse_log_Jacobian(
        mapping::SuperHeavyReference{<:Any, <:Any, T}, 
        u::PSDdata{T}
    ) where {T<:Number}
    # inverse function theorem
    return -SMT.marginal_log_Jacobian(mapping, SMT.marginal_pullback(mapping, u))
end