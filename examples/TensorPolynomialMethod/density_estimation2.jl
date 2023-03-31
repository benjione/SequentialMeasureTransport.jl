using PSDModels
using LinearAlgebra
using Plots
using ApproxFun
# amount data to generate
N = 250

# generate data according to some distribution
X1 = randn(N) * 0.75 .+ 1.5
X2 = randn(N) * 0.75 .- 1.5
X = [X1; X2]
pdf_X1(x) = 1/(2*0.75*π)^(0.5) * exp(-(x - 1.5)^2/(2*0.75))
pdf_X2(x) = 1/(2*0.75*π)^(0.5) * exp(-(x + 1.5)^2/(2*0.75))
pdf_X(x) = (pdf_X1(x) + pdf_X2(x)) / 2

# Create an empty model
model = PSDModel(Hermite(), :trivial, 20)

# use log-likelihood loss
loss(Z) = -(1/length(Z)) * sum(log.(Z))

# minimize loss
minimize!(model, loss, X, trace=true, λ_1=0.001, normalization_constraint=true)

margin = marginalize_orth_measure(model, 1, measure_scale=1.0)
# model = model * (1/margin)

real_fac = PSDModels.integrate(model, -15..15, amount_quadrature_points=50)
model = model * (1/real_fac)
# plot all
domx = range(-15, 15, length=400)
plot(domx, model.(domx), label="fitted model")
plot!(X, model.(X), seriestype=:scatter, label="data points")

# Plot the model
dom_x = range(-4, 4, length=400)
plot(dom_x, model.(dom_x).*exp.(-domx.^2 ./2.0), label="\$f_A(x)\$")
plot!(dom_x, pdf_X.(dom_x), label="\$f(x)\$")
# plot!(X, Y, seriestype=:scatter, label="data")
# plot!(model.X, model.(model.X), seriestype=:scatter, label="data points")
savefig("density_estimation2.pdf")