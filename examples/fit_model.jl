using PSDModels
using KernelFunctions
using LinearAlgebra
using Plots

f(x) = 2*(x-0.5)^2 * (x+0.5)^2

# amount data to generate
N = 10

# Generate some points
support = collect(range(-1, 1, length=N))

# Create an empty model
k = MaternKernel(ν=1.0)
model = PSDModel(k, support)

# generate some data
X = rand(2*N) * 2 .-1
Y = f.(X)

# fit the model
fit!(model, X, Y)

# Plot the model
dom_x = range(-1, 1, length=100)
plot(dom_x, model.(dom_x), label="model fitted")
plot!(dom_x, f.(dom_x), label="f(x)")
plot!(X, Y, seriestype=:scatter, label="data")
plot!(model.X, model.(model.X), seriestype=:scatter, label="support points")