using PSDModels
using KernelFunctions
using LinearAlgebra
using Plots
using DomainSets

f(x) = 2*(x-0.5)^2 * (x+0.5)^2

f_int(x) = 0.125*x + 0.4*x^5 - (1/3)*x^3

# amount data to generate
N = 10

# Generate some data
X = collect(range(-1, 1, length=N))
Y = f.(X)

# Create a model
k = MaternKernel(ν=1.0)
# note that gradients from the direct model are not smooth
model = PSDModel(X, Y, k; solver=:gradient_descent)

# Plot the model and integral
dom_x = range(-1, 1, length=100)
dom_to_x(x) = Interval(0, x)

plot(dom_x, model.(dom_x), ylims=(-1,2), label="model")
plot!(dom_x, f.(dom_x), label="f(x)")
plot!(X, Y, seriestype=:scatter, label="data")

plot(dom_x, integral.(Ref(model), dom_to_x.(dom_x)), label="model integral")
plot!(dom_x, f_int.(dom_x), label="F(x)")