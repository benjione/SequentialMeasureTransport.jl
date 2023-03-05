using PSDModels
using LinearAlgebra
using ApproxFun
using Plots

f(x) = 2*(x-0.5)^2 * (x+0.5)^2

# order of feature map
N = 20

# Create an empty model
FM_funcs = [Fun(Chebyshev(-1..1), [zeros(k); 1.0]) for k in 0:N]
Φ(x) = map(f -> f(x), FM_funcs)
model = PSDModel(Φ, N+1)

# generate some data
X = rand(20*N) * 2 .-1
Y = f.(X)

# fit the model
fit!(model, X, Y, trace=true)

# Plot the model
dom_x = range(-1, 1, length=100)
plot(dom_x, model.(dom_x), label="model fitted")
plot!(dom_x, f.(dom_x), label="f(x)")
plot!(X, Y, seriestype=:scatter, label="data")