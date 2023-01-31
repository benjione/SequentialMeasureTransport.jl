using PSDModels
using KernelFunctions
using LinearAlgebra
using Plots
import ForwardDiff as FD

f(x) = 2*(x-0.5)^2 * (x+0.5)^2

# amount data to generate
N = 10

# Generate some data
X = collect(range(-1, 1, length=N))
Y = f.(X)

# Create a model
k = MaternKernel(ν=1.0)
# note that gradients from the direct model are not smooth
model = PSDModel(X, Y, k; solver=:gradient_descent)

# Plot the model and gradient
dom_x = range(-1, 1, length=100)
plot(dom_x, model.(dom_x), ylims=(-1,2), label="model")
plot!(dom_x, f.(dom_x), label="f(x)")
plot!(X, Y, seriestype=:scatter, label="data")

plot(dom_x, gradient.(Ref(model), dom_x), label="model gradient")
plot!(dom_x, FD.derivative.(f, dom_x), label="f'(x)")