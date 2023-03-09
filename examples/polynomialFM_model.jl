using PSDModels
using LinearAlgebra
using ApproxFun
using Plots


f(x) = 2*(x-0.5)^2 * (x+0.5)^2
f(x) = 2*(x[2]-0.5)^2 * (x[2]+0.5)^2 + 2*(x[1]-0.5)^2 * (x[1]+0.5)^2

# Create an empty model
model = PSDModel(Chebyshev(), 20)

# generate some data
X = rand(200) * 2 .-1
Y = f.(X)

# fit the model
fit!(model, X, Y, trace=true)

# Plot the model
dom_x = range(-1, 1, length=100)
plot(dom_x, model.(dom_x), label="model fitted")
plot!(dom_x, f.(dom_x), label="f(x)")
plot!(X, Y, seriestype=:scatter, label="data")