using PSDModels
using LinearAlgebra
using ApproxFun
using Plots



f(x) = 2*(x[2]-0.5)^2 * (x[2]+0.5)^2 + 2*(x[1]-0.5)^2 * (x[1]+0.5)^2
f_int(x) = @error "todo"

# Create an empty model
model = PSDModel(Chebyshev()^2, 20)

# generate some data
X = [(rand(2) * 2 .- 1) for i in 1:300]
Y = f.(X)

# fit the model
fit!(model, X, Y, trace=true)

model_integrated = integral(model, [1, 0])

# Plot the model
dom_x = range(-1, 1, length=100)
plot(dom_x, model.(dom_x), label="model fitted")
plot!(dom_x, f.(dom_x), label="\$f(x)\$")
plot!(dom_x, model_integrated.(dom_x), label="\$\\int f(x)\$")
plot!(X, Y, seriestype=:scatter, label="data")