using PSDModels
using KernelFunctions
using LinearAlgebra
using Plots

f(x) = 2*(x[2]-0.5)^2 * (x[2]+0.5)^2 + 2*(x[1]-0.5)^2 * (x[1]+0.5)^2

# amount data to generate
N = 10

# Generate some points
support = [[x,y] for x in range(-1, 1, length=N), 
                     y in range(-1, 1, length=N)]
support = reshape(support, length(support))

# Create an empty model
k = MaternKernel(ν=1.0)
model = PSDModel(k, support)

# generate some data
X = [(rand(2) * 2 .- 1) for i in 1:2*N^2]
Y = f.(X)

# fit the model
fit!(model, X, Y)

# Plot the model
dom_x = range(-1, 1, length=100)
dom_y = range(-1, 1, length=100)
dom_xy = [[x,y] for x in range(-1,1,100), y in range(-1,1,100)]

plt1 = contour(dom_x, dom_y, model.(dom_xy), 
            title="fitted")
plt2 = contour(dom_x, dom_y, f.(dom_xy), 
            title="f(x)")

plot(plt1, plt2, layout=(1,2))