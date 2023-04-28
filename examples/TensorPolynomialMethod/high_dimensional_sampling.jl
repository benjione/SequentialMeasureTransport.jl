using PSDModels
using LinearAlgebra
using Plots
# using Distributions
using ApproxFun
import Convex as con
using NNlib # for relu
using Hypatia

σ_v = 5.0
pdf_X1(x) = 1/(2*σ_v^2*π)^(1) * exp(-norm(x - [1.5, 1.5])^2/(2*σ_v^2))
pdf_X2(x) = 1/(2*σ_v^2*π)^(1) * exp(-norm(x + [1.5, 1.5])^2/(2*σ_v^2))
pdf_X(x) = (pdf_X1(x) + pdf_X2(x)) / 2
pdf_X(x) = (20^2 -0.1(x[1]^2 + x[2]^2))

domx = collect(range(-10, 10, 100))
domy = collect(range(-10, 10, 100))

surface(domx, domy, (x,y)->pdf_X([x,y]))

model = PSDModel(Legendre(-10..10)^2, :downward_closed, 3)
model = PSDModel(Legendre(-10..10)^2, :trivial, 15)

X = [(rand(2).-0.5)*20.0 for _=1:400]
loss(Z) = (1/length(Z)) * mapreduce((x,z)->pdf_X(x)^2 * con.invpos(z), +, X, Z)
loss_KL(Z) = -(1/length(Z)) * mapreduce((x,z)->pdf_X(x) * log(z), +, X, Z)
loss_Hell(Z) = (1/length(Z)) * mapreduce((x,z)->(sqrt(pdf_X(x)) - sqrt(z))^2, +, X, Z)
# loss_stable(Z) = -(1/length(Z)) * mapreduce((x,z)->(log(pdf_X(x)/z) - mapreduce(log), +, X, Z)
loss_KL2(Z) = -(1/length(Z)) * mapreduce((x,z)->log(z/pdf_X(x)) * pdf_X(x), +, X, Z)
loss_chi(Z) = (1/length(Z)) * mapreduce((x,z)->pdf_X(x)^2 * con.invpos(z+0.0001) + 2*pdf_X(x)+ z, +, X, Z)
function loss_stable(Z)
    E_G = 1/length(Z) * mapreduce((x,z)->log(pdf_X(x)/z), +, X, Z)
    sq_plus(a) = max(0.0, a)^2
    # sq_plus(a) = relu(a)^2
    return 1/length(Z) * mapreduce((x,z)->sq_plus(log(pdf_X(x)/z) - E_G), +, X, Z)
end

combined_loss(Z) = loss_stable(Z) + loss_KL(Z)
X2 = [(rand(2).-0.5)*20.0 for _=1:4000]
X_t = [X; X2]
loss_KL_reg(Z) = loss_KL(Z[1:length(X)])

# fit!(model, X, pdf_X.(X), trace=true, maxit=10000, 
#         normalization_constraint=false, λ_2=0.0, λ_1=0.0,
#         optimizer=Hypatia.Optimizer)

con.emit_dcp_warnings() = false
minimize!(model, loss_chi, X, trace=true, 
        λ_2=0.0, λ_1=0.0, maxit=10000,
        normalization_constraint=false, convex=true,
        optimizer=Hypatia.Optimizer
        )

scatter([x[1] for x in X], [x[2] for x in X], pdf_X.(X), label="data points")
# plot(
#     surface(domx, domy, (x,y)->model([x,y])),
#     surface(domx, domy, (x,y)->f([x,y]))
# )

PSDModels.normalize!(model)
plot(
    surface(domx, domy, (x,y)->model([x,y])),
    surface(domx, domy, (x,y)->pdf_X([x,y]))
)