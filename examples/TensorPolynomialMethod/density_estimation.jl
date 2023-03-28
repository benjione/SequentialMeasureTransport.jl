using PSDModels
using LinearAlgebra
using Plots
using ApproxFun
# amount data to generate
N = 400

# generate data according to some distribution
X = randn(N) * 0.75 .+ 0.5
pdf_X(x) = 1/(2*0.75*π)^(0.5) * exp(-(x - 0.5)^2/(2*0.75))

# Create an empty model
model = PSDModel(Chebyshev(-15..15), :trivial, 30)

# use log-likelihood loss
loss(Z) = -(1/length(Z)) * sum(log.(Z))

# minimize loss
minimize!(model, loss, X, trace=true, λ_1=0.1)

margin = marginalize_orth_measure(model, 1, measure_scale=1.0)
model = model * (1/margin)

real_fac = PSDModels.integrate(model, -15..15, amount_quadrature_points=50)
model = model * (1/real_fac)
# plot all
domx = range(-15, 15, length=400)
plot(domx, model.(domx), label="fitted model")
plot!(X, model.(X), seriestype=:scatter, label="data points")

# Plot the model
dom_x = range(-2, 2.5, length=100)
plot(dom_x, model.(dom_x), label="fitted model")
plot!(dom_x, pdf_X.(dom_x), label="f(x)")
# plot!(X, Y, seriestype=:scatter, label="data")
# plot!(model.X, model.(model.X), seriestype=:scatter, label="data points")
