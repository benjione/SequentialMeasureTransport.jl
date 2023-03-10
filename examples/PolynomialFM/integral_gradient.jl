using PSDModels
using LinearAlgebra
using ApproxFun
using Plots


f(x) = 2*(x-0.5)^2 * (x+0.5)^2
f_int(x) = 0.1250000000*x + 0.4000000000*x^5 - 0.3333333333*x^3

# Create an empty model
model = PSDModel(Chebyshev(), 20)

# generate some data
X = rand(400) * 2 .-1
Y = f.(X)

# fit the model
fit!(model, X, Y, trace=true, maxit=1000)

model_integrated = integral(model, [1])

# Plot the model
dom_x = range(-1, 1, length=100)
plot(dom_x, model.(dom_x), label="model fitted")
plot!(dom_x, f.(dom_x), label="\$f(x)\$")
plot!(dom_x, model_integrated.(dom_x), label="\$\\int f(x)\$")
plot!(X, Y, seriestype=:scatter, label="data")