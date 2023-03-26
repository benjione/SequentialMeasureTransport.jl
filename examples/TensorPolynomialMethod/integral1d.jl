using PSDModels
using LinearAlgebra
using ApproxFun
using Plots

f(x) = x^2
f_int(x) = 1/3 * x^3

# Create an empty model using Legendre polynimials, since they are orthogonal
# to the Lebesgue measure.
model = PSDModel(Legendre(), :trivial, 5)

# generate some data
X = collect(range(-1, 1, length=100))
Y = f.(X)

# fit the model
fit!(model, X, Y, trace=true)

# give integral over dim 1
int_model = integral(model, 1)

# Plot the model
dom_x = range(-1, 1, length=100)

plt1 = plot(dom_x, model.(dom_x), 
            title="model")
plot!(plt1, dom_x, f.(dom_x), label="x^2")

# Plot the margin
plt2 = plot(dom_x, int_model.(dom_x), label="int")
plot!(plt2, dom_x, f_int.(dom_x), label="1/3 x^3")

plot(plt1, plt2, layout=(1,2))