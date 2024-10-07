using SequentialMeasureTransport
using ApproxFun
import SequentialMeasureTransport as SMT
using LinearAlgebra
using Distributions
using Plots
using MosekTools

include("helpers.jl")


## define a cost function
c(x) = (x[1] - x[2])^2

## define the marginals
p = MixtureModel(Normal[
    Normal(0.3, 0.1),
    Normal(0.7, 0.1)
])
q = Normal(0.5, 0.2)



# model = SMT.PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 4)
# _smp = SMT.Sampler(model)
smp_list = SMT.ConditionalMapping{2, 0, Float64}[]
ϵ_list = [1.0, 0.5, 0.3, 0.15, 0.08]

for k=1:5
    XY = rand(2, 1500)
    X = rand(1, 500)
    Y = rand(1, 500)

    model = SMT.PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 4)

    prec = if k > 1
        SMT.CondSampler(smp_list[1:k-1], nothing)
    else
        nothing
    end

    SMT.Statistics.entropic_OT!(model, c, x->pdf(p, x)[1], x->pdf(q, x)[1],
        ϵ_list[k], eachcol(XY), trace=true,
        X=eachcol(X), Y=eachcol(Y),
        optimizer=Mosek.Optimizer,
        set_start_value=false,
        preconditioner=prec,
        # use_preconditioner_cost=use_prec,
        normalization=true)

    normalize!(model)
    push!(smp_list, SMT.Sampler(model))
end

rng = 0.0:0.01:1.0
_smp = SMT.CondSampler(smp_list[1:4], nothing)
contourf(rng, rng, (x, y) -> pdf(_smp, [x, y]), alpha=1.0, label="transported model")

M_sink = compute_Sinkhorn(rng, x->pdf(p, x)[1], 
            x->pdf(q, x)[1], c, 0.15)
contourf(rng, rng, M_sink')

contour(
    rng, rng, (x, y) -> 0.5*pdf(_smp, [x, y]), levels=5
)
contour!(rng, rng, M_sink', levels=5)

rng = 0.01:0.01:0.99
plot(rng, x->left_pdf(_smp, x))
plot!(rng, x->pdf(p, x)[1])

plot(rng, x->right_pdf(_smp, x))
plot!(rng, x->pdf(q, x)[1])