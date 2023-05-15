using PSDModels
using PSDModels.Statistics
using LinearAlgebra
using ApproxFun
using Plots
using Distributions
# using Hypatia

# dicontinuous Distributions
# f(x) = norm(x) ≤ 2 ? 1/(4*π) : 0.0
f1(x) = pdf(Normal(0, 1.5),x[2]+4.0) * pdf(Normal(x[2]+4.0, 0.5),0.2*(x[1])^2)
f2(x) = pdf(Normal(0, 1.1),x[2]-2.0) * pdf(Normal(x[2]-2.0, 0.5),0.5*(x[1])^2)
f(x) = (f1(x) + f2(x) )/ 2.0

a = -8
b = 8
rng = range(a, b, length=100)

contour(rng, rng, (x,y)->f([x,y]), levels=20, c=:blues, label="true")

model_sr = PSDModel(Legendre(a..b)^2, :downward_closed, 6)
sar = SelfReinforcedSampler(f, model_sr, 4, :Chi2, trace=true,
                            ϵ=1e-9, λ_2=0.0, λ_1=0.0,
                            min_IRLS_iter=2, max_IRLS_iter=2,
                            relaxation_method=:blurring,
                            max_blur=1.5, N_MC_blurring=50)

pb_f = PSDModels.pushforward_pdf_function(sar, x->1/(b-a)^2)

rng2 = range(a+0.5, b-0.5, length=100)
plot(
    contour(rng2, rng2, (x,y)->pb_f([x,y]), levels=20, c=:blues, label="true"),
    contour(rng2, rng2, (x,y)->f([x,y]), levels=20, c=:blues, label="true"),
)

samples = PSDModels.sample(sar, 5000, threading=true)
histogram2d([s[1] for s in samples], [s[2] for s in samples], 
            nbins=100, c=:blues, label="data", normalize=true)

E = PSDModels.Statistics.expectation(sar)
println("Expectation: ", E)

Cov = PSDModels.Statistics.covariance(sar)
println("Covariance: ", Cov)