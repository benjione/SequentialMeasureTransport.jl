using PSDModels
using LinearAlgebra
using ApproxFun
using Plots

f(x) = 1/(2.0 * π) * exp(-x[1]^2/2) * exp(-x[2]^2/2)
# result of ∫_{-1}^{1} f(x_1, x_2) dx_1
f_marg(x) = (0.85/π) * exp(-x^2/2)

# Create an empty model using Legendre polynimials, since they are orthogonal
# to the Lebesgue measure.
model = PSDModel(Legendre()^2, :trivial, 25)

# generate some data
X = [(rand(2) * 2 .- 1) for i in 1:300]
Y = f.(X)

# fit the model
fit!(model, X, Y, trace=true)

# marginalize the model
model_marginalized = marginalize_orth_measure(model, 1)

# Plot the model
dom_x = range(-1, 1, length=100)
dom_y = range(-1, 1, length=100)
dom_xy = [[x,y] for x in range(-1,1,100), y in range(-1,1,100)]

plt1 = contour(dom_x, dom_y, model.(dom_xy), 
            title="fitted")

# Plot the margin
dom_x = range(-1, 1, length=100)
plt2 = plot(dom_x, model_marginalized.(dom_x), label="margin")
plot!(plt2, dom_x, f_marg.(dom_x), label="exp(-x^2/2)")

plot(plt1, plt2, layout=(1,2))