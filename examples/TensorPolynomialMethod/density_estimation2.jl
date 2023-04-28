using PSDModels
using PSDModels.Statistics # for ML estimation
using LinearAlgebra
using Plots
using ApproxFun
# using Hypatia
# amount data to generate
N = 200

# generate data according to some distribution
X1 = randn(N) * 0.75 .+ 1.5
X2 = randn(N) * 0.75 .- 1.5
X = [X1; X2]
pdf_X1(x) = 1/(2*0.75*π)^(0.5) * exp(-(x - 1.5)^2/(2*0.75))
pdf_X2(x) = 1/(2*0.75*π)^(0.5) * exp(-(x + 1.5)^2/(2*0.75))
pdf_X(x) = (pdf_X1(x) + pdf_X2(x)) / 2

X_chi = (rand(400) .-0.5) * 10.0

# Create an empty model
model_KL = PSDModel(Legendre(-5..5), :downward_closed, 7)
model_chi = PSDModel(Legendre(-5..5), :downward_closed, 7)

# minimize loss
ML_fit!(model_KL, X, trace=true, λ_2=1e-8)

Chi2_fit!(model_chi, X_chi, pdf_X.(X_chi), trace=true, λ_2=1e-8)

# plot all
# domx = range(-15, 15, length=400)
# plot(domx, model.(domx), label="fitted model")
# plot!(X, model.(X), seriestype=:scatter, label="data points")

# Plot the model
dom_x = range(-5, 5, length=400)
plot(dom_x, model_KL.(dom_x), label="\$f_A(x)\$")
plot!(dom_x, pdf_X.(dom_x), label="\$f(x)\$")
plot!(dom_x, model_chi.(dom_x), label="\$f_\\chi(x)\$")
