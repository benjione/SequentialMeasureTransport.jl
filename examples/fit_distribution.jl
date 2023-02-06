using PSDModels
using KernelFunctions
using LinearAlgebra
using Plots
using DomainSets
# amount data to generate
N = 100

# generate data according to some distribution
X = randn(N) * 0.75 .+ 0.5
pdf_X(x) = 1/(2*0.75*π)^(0.5) * exp(-(x - 0.5)^2/(2*0.75))

# Create an empty model
k = MaternKernel(ν=1.0)
model = PSDModel(k, X)

# use log-likelihood loss
loss(Z) = -(1/length(Z)) * sum(log.(Z))

# minimize loss
minimize!(model, loss, X, trace=true)

# normalize the model (implement as a constraint into the minimization process, 
#        see https://juliafirstorder.github.io/ProximalOperators.jl/stable/functions/#ProximalOperators.IndAffine)
model = model * (1/integral(model, -10..10, amount_quadrature_points=50))

# Plot the model
dom_x = range(-2, 2.5, length=100)
plot(dom_x, model.(dom_x), label="fitted model")
plot!(dom_x, pdf_X.(dom_x), label="f(x)")
# plot!(X, Y, seriestype=:scatter, label="data")
plot!(model.X, model.(model.X), seriestype=:scatter, label="data points")

savefig("fig/fit_distribution.png")