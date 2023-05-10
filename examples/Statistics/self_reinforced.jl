using PSDModels
using PSDModels.Statistics
using ApproxFun
using Plots
using Distributions
# using Hypatia

mean = [0.2,0.2]
C = [1.5 0.0; 0.0 1.5]
d = MvNormal(mean, C)
x = rand(d, 1000)
f(x) = pdf(d,x)

histogram2d(x[1,:], x[2,:], nbins=50, c=:blues, label="data")

a = -4
b = 4
rng = range(a, b, length=100)

surface(rng, rng, (x,y)->f([x,y]), levels=20, c=:blues, label="true")

model = PSDModel(Legendre(a..b)^2, :downward_closed, 8)

sample_X = rand(2000, 2) .* (b-a) .+ a
sample_Y = f.(eachrow(sample_X))
Chi2_fit!(model, collect(eachrow(sample_X)),sample_Y, trace=true,
            λ_2=0.0, λ_1=0.0, ϵ=1e-6,
            # optimizer=Hypatia.Optimizer
            )

PSDModels.normalize!(model)
surface(rng, rng, (x,y)->model([x,y]), levels=20, c=:blues, label="ML estimation")


model_sr = PSDModel(Legendre(a..b)^2, :downward_closed, 3)
sar = SelfReinforcedSampler(f, model_sr, 4, :Chi2, trace=true,
                            ϵ=1e-6, λ_2=0.0, λ_1=0.0)

pb_f = PSDModels.pushforward_pdf_function(sar, x->16.0^4)

rng2 = range(-4, 4, length=100)
surface(rng2, rng2, (x,y)->pb_f([x,y]), levels=20, c=:blues, label="true")

E = PSDModels.Statistics.expectation(model)
println("Expectation: ", E)

E = PSDModels.Statistics.expectation(sar)
println("Expectation: ", E)

Cov = PSDModels.Statistics.covariance(model)
println("Covariance: ", Cov)

Cov = PSDModels.Statistics.covariance(sar)
println("Covariance: ", Cov)