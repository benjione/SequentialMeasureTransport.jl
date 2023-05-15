using PSDModels
using LinearAlgebra
using ApproxFun
using Plots
# using Hypatia

# f(x) = 2*(x[2]-0.5)^2 * (x[2]+0.5)^2 + 2*(x[1]-0.5)^2 * (x[1]+0.5)^2
f(x) = exp((-x[1]^2 - x[2]^2)/0.1)

# Create an empty model
model = PSDModel(Chebyshev()^2, :downward_closed, 1)
model_direct = PSDModel(Chebyshev()^2, :downward_closed, 4)
# generate some data
X = [(rand(2) * 2 .- 1) for i in 1:1000]
Y = f.(X)

# fit the model
fit!(model_direct, X, Y, trace=true)
new_model, losses = PSDModels.greedy_fit(model, X, Y, 
                ones(length(Y)), trace=false, greedy_trace=true,
                max_greedy_iterations=20, greedy_conv_tol=1.1)


# Plot the loss
scatter([i[1] for i in new_model.Φ.ten.index_list], [i[2] for i in new_model.Φ.ten.index_list])
plot(losses, yscale=:log10)
# Plot the model
dom_x = range(-1, 1, length=100)
dom_y = range(-1, 1, length=100)
dom_xy = [[x,y] for x in range(-1,1,100), y in range(-1,1,100)]

plt1 = contour(dom_x, dom_y, new_model.(dom_xy), 
            title="fitted")
plt2 = contour(dom_x, dom_y, f.(dom_xy), 
            title="f(x)")

plot(plt1, plt2, layout=(1,2))

surface(dom_x, dom_y, new_model.(dom_xy), 
            title="fitted greedy")
surface(dom_x, dom_y, model_direct.(dom_xy), 
            title="fitted")