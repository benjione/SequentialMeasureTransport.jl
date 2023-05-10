using PSDModels
using PSDModels.Statistics
using ApproxFun
using Plots
using Distributions
using Hypatia

mean = [0.,0.]
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

ML_fit!(model, collect(eachcol(x)), trace=true,
        λ_2=0.0, λ_1=1.0,
        # optimizer=Hypatia.Optimizer()
        )

sample_X = rand(2000, 2) .* (b-a) .+ a
sample_Y = f.(eachrow(sample_X))
Chi2_fit!(model, collect(eachrow(sample_X)),sample_Y, trace=true,
            λ_2=0.0, λ_1=0.0, ϵ=1e-6,
            # optimizer=Hypatia.Optimizer
            )

PSDModels.normalize!(model)
surface(rng, rng, (x,y)->model([x,y]), levels=20, c=:blues, label="ML estimation")


E = PSDModels.Statistics.expectation(model)
println("Expectation: ", E)

Cov = PSDModels.Statistics.covariance(model)
println("Covariance: ", Cov)