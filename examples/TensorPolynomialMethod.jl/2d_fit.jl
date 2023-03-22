using PSDModels
using LinearAlgebra
using ApproxFun
using Plots

f(x) = 2*(x[2]-0.5)^2 * (x[2]+0.5)^2 + 2*(x[1]-0.5)^2 * (x[1]+0.5)^2

# Create an empty model
model = PSDModel(Chebyshev()^2, :trivial, 20)

# generate some data
X = [(rand(2) * 2 .- 1) for i in 1:300]
Y = f.(X)

# fit the model
fit!(model, X, Y, trace=true)

# Plot the model
dom_x = range(-1, 1, length=100)
dom_y = range(-1, 1, length=100)
dom_xy = [[x,y] for x in range(-1,1,100), y in range(-1,1,100)]

plt1 = contour(dom_x, dom_y, model.(dom_xy), 
            title="fitted")
plt2 = contour(dom_x, dom_y, f.(dom_xy), 
            title="f(x)")

plot(plt1, plt2, layout=(1,2))