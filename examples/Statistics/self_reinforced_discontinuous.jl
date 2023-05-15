using PSDModels
using PSDModels.Statistics
using LinearAlgebra
using ApproxFun
using Plots
using Distributions
# using Hypatia

# dicontinuous Distributions
f(x) = norm(x) ≤ 2 ? 1/(4*π) : 0.0

a = -8
b = 8
rng = range(a, b, length=100)

contour(rng, rng, (x,y)->f([x,y]), levels=20, c=:blues, label="true")

model_sr = PSDModel(Legendre(a..b)^2, :downward_closed, 5)
sar = SelfReinforcedSampler(f, model_sr, 4, :Chi2, trace=true,
                            ϵ=1e-6, λ_2=0.0, λ_1=0.1,
                            relaxation_method=:blurring,
                            max_blur=1.5, N_MC_blurring=50)

pb_f = PSDModels.pushforward_pdf_function(sar, x->1/(b-a)^2)

rng2 = range(a+0.5, b-0.5, length=100)
plot(
    surface(rng2, rng2, (x,y)->pb_f([x,y]), levels=20, c=:blues, label="true"),
    surface(rng2, rng2, (x,y)->f([x,y]), levels=20, c=:blues, label="true"),
)

samples = PSDModels.sample(sar, 5000, threading=true)
histogram2d([s[1] for s in samples], [s[2] for s in samples], 
            nbins=100, c=:blues, label="data", normalize=true)

E = PSDModels.Statistics.expectation(sar)
println("Expectation: ", E)

Cov = PSDModels.Statistics.covariance(sar)
println("Covariance: ", Cov)