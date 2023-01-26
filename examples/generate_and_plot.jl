using PSDModels
using KernelFunctions
using LinearAlgebra
using Plots

f(x) = 2*(x-0.5)^2 * (x+0.5)^2

# amount data to generate
N = 100

# Generate some data
X = collect(range(-1, 1, length=N))
Y = f.(X)

# Create a model
k = MaternKernel(ν=1.0)
model = PSDModel(X, Y, k)

# Plot the model
dom_x = range(-1, 1, length=100)
plot(dom_x, model.(dom_x), label="model")
plot!(dom_x, f.(dom_x), label="true function")
plot!(X, Y, seriestype=:scatter, label="data")