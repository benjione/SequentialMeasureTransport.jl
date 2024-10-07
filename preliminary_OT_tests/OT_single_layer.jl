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


### First, we work on the bounded domain [0, 1]^2

model = SMT.PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 5)

# we generate some random points to estimate the cost function
XY = rand(2, 2000)

## by default, the marginals are satisfied by the marginals of XY.
## we can also specify the marginal point constraints explicitly.
X = rand(1, 1000)
Y = rand(1, 1000)

SMT.Statistics.entropic_OT!(model, c, x->pdf(p, x)[1], x->pdf(q, x)[1],
                            0.3, eachcol(XY), trace=true,
                            X=eachcol(X), Y=eachcol(Y),
                            optimizer=Mosek.Optimizer,
                            set_start_value=false,)

normalize!(model)

rng = 0.0:0.01:1.0
contourf(rng, rng, (x, y) -> model([x, y]), alpha=1.0, label="transported model")


### Let us now demonstrate to work on the unbounded domain R^2
# first, we need a preconditioner for the unbounded domain

ref_map = SMT.ReferenceMaps.AlgebraicReference{2, Float64}()

model = SMT.PSDModel(Legendre(0..1)^2, :downward_closed, 5)

## define the marginals
p = MixtureModel(Normal[
    Normal(-1.0, 0.3),
    Normal(1.0, 0.3)
])
q = Normal(0.0, 0.5)


## Let us generate some random points in R^2.
## Best is, we first sample in [0, 1]^2 and then apply the inverse of the reference map.
XY_R2 = SMT.pullback.(Ref(ref_map), eachcol(XY))

X_R_Y_R = SMT.pullback.(Ref(ref_map), eachcol([X ; Y]))

X_R = [[x[1]] for x in X_R_Y_R]
Y_R = [[x[2]] for x in X_R_Y_R]

rng = 0:0.01:1.0
cost_pb = SMT.pushforward(ref_map, c)
marg_dist_pb = SMT.pushforward(ref_map, x->pdf(p, x[1]) * pdf(q, x[2]))
contourf(rng, rng, (x, y) -> cost_pb([x, y]), alpha=1.0, label="cost function")

SMT.Statistics.entropic_OT!(model, c, x->pdf(p, x)[1], x->pdf(q, x)[1],
                            0.3, eachcol(XY), trace=true,
                            X=X_R, Y=Y_R,
                            optimizer=Mosek.Optimizer,
                            # preconditioner=precond,
                            reference=ref_map, # this is a diagonal map
                            set_start_value=false,)

normalize!(model)
rng = 0.0:0.01:0.0
model_R = SMT.pushforward(ref_map, x->model(x))
contour(rng, rng, (x, y) -> model([x, y]), alpha=1.0, label="transported model")

plot(rng, x->model([x, 0.5]), label="x-marginal")