using SequentialMeasureTransport
using ApproxFun
import SequentialMeasureTransport as SMT
using LinearAlgebra
using Distributions
using Plots
using MosekTools

include("helpers.jl")


## define a cost function
_d = 1
c(x, y) = norm(x - y)^2
c(x) = norm(x[1:_d] - x[_d+1:end])^2

## define the marginals
p = MixtureModel(Normal[
    Normal(0.3, 0.1),
    Normal(0.7, 0.1)
])
q = Normal(0.5, 0.2)


### First, we work on the bounded domain [0, 1]^2

ϵ = 0.5

# model = SMT.PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 4)
# _smp = SMT.Sampler(model)
smp_list = SMT.ConditionalMapping{2, 0, Float64}[]
# λ_marg_list = 1e12 * ones(20)

for k=1:2
    XY = rand(2, 2000)
    X = rand(1, 1000)
    Y = rand(1, 1000)

    model = SMT.PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 4)

    prec = if k > 1
        SMT.CondSampler(smp_list[1:k-1], nothing)
    else
        nothing
    end
    use_prec = if k > 1
        true
    else
        false
    end

    SMT.Statistics.entropic_OT!(model, c, x->pdf(p, x)[1], x->pdf(q, x)[1],
        ϵ, eachcol(XY), trace=true,
        X=eachcol(X), Y=eachcol(Y),
        optimizer=Mosek.Optimizer,
        set_start_value=false,
        preconditioner=prec,
        use_preconditioner_cost=use_prec,
        normalization=true)

    normalize!(model)
    push!(smp_list, SMT.Sampler(model))
end

rng = 0.0:0.01:1.0
_smp = SMT.CondSampler([smp_list[1:2]...], nothing)
contourf(rng, rng, (x, y) -> pdf(_smp, [x, y]), alpha=1.0, label="transported model")

M_sink = compute_Sinkhorn(rng, x->pdf(p, x)[1], 
            x->pdf(q, x)[1], c, 0.05)
contourf(rng, rng, M_sink')

contour(
    rng, rng, (x, y) -> pdf(_smp, [x, y]), levels=5
)
contour!(rng, rng, M_sink', levels=5)

plot(rng, x->left_pdf(_smp, x))
plot!(rng, x->pdf(p, x)[1])

plot(rng, x->right_pdf(_smp, x))
plot!(rng, x->pdf(q, x)[1])

compute_Sinkhorn_distance(c, _smp, N=3000)
compute_Sinkhorn_distance(c, M_sink, rng)

distance_list = Float64[]
ϵ_list = Float64[]
rng = 0:0.005:1
for k=1:10
    ϵ = 0.5^(k)
    push!(ϵ_list, ϵ)
    M_sink = compute_Sinkhorn(rng, x->pdf(p, x)[1], 
                x->pdf(q, x)[1], c, ϵ)
    push!(distance_list, compute_Sinkhorn_distance(c, M_sink, rng))
end
plot(ϵ_list, distance_list, m=:o, label="Sinkhorn distance", xscale=:log10)


distance_chained_list = Float64[]
for k=1:length(smp_list)
    _smp = SMT.CondSampler(smp_list[1:k], nothing)
    push!(distance_chained_list, compute_Sinkhorn_distance(c, _smp, N=5000))
end
plot(1:length(smp_list), distance_chained_list, m=:o, label="Chained Sinkhorn distance")