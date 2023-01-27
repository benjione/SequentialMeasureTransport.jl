using PSDModels
using KernelFunctions
using LinearAlgebra
using Plots

f(x) = 2*(x-0.5)^2 * (x+0.5)^2

# amount data to generate
N = 100

# Generate some data
X = rand(N) * 2 .- 1
Y = f.(X)

# Create a model
k = MaternKernel(ν=1.0)
model_direct = PSDModel(X, Y, k, solver=:direct)
model_gd = PSDModel(X, Y, k, solver=:gradient_descent)

# Plot the model
dom_x = range(-1, 1, length=100)
plot(dom_x, model_direct.(dom_x), ylims=(0,1.5), label="model direct")
plot!(dom_x, model_gd.(dom_x), label="model gradient descent")
plot!(dom_x, f.(dom_x), label="f(x)")
plot!(X, Y, seriestype=:scatter, label="data")