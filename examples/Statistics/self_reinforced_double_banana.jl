using PSDModels
using PSDModels.Statistics
using LinearAlgebra
using ApproxFun
using Plots
using Distributions
using Hypatia

# dicontinuous Distributions
# f(x) = norm(x) ≤ 2 ? 1/(4*π) : 0.0
f1(x) = pdf(Normal(0, 0.75),x[2]+2.0) * pdf(Normal(x[2]+2.0, 0.25),0.2*(x[1])^2)
f2(x) = pdf(Normal(0, 0.55),x[2]-1.0) * pdf(Normal(x[2]-1.0, 0.25),0.5*(x[1])^2)
f(x) = (f1(x) + f2(x) )/ 2.0
# f(x) = pdf(MvNormal([0,0], diagm([2.0, 3.0])), x)

a = -5
b = 5
rng = range(a, b, length=150)

contour(rng, rng, (x,y)->f([x,y]), levels=20, c=:blues, label="true", colorbar=nothing)

sar = SelfReinforcedSampler(x->f(x),
        PSDModel(Legendre()^2, :downward_closed, 10, mapping=:algebraicOMF),
        3, :Chi2, trace=true,
        relaxation_method=:algebraic,
        λ_1=0.0, λ_2=0.0,
        min_IRLS_iter=1, max_IRLS_iter=5,
        N_samples=2000,
        reference_map=PSDModels.GaussianReference{2, Float64}(3.0),
        optimizer=Hypatia.Optimizer,
        algebraic_base=3.0,
)   

plot(
    contour(rng, rng, (x,y)->pdf(sar, [x,y]), levels=20, c=:vik, label="true"),
    contour(rng, rng, (x,y)->f([x,y]), levels=20, c=:vik, label="true"),
)
# contour(rng, rng, (x,y)->pullback_pdf_function([x,y]), levels=20, c=:blues, label="true")
import PSDModels.Plotting as PSDplot
PSDplot.plot_sampler2D(sar, [x->f(x)^(1/9), x->f(x)^(1/3), x->f(x)]; domain=(-6, 6),
                N_plot=150, colorbar=nothing, levels=10, c=:vik, showaxis=false,
                grid=false, single_plot=false, titles=false, savefig_path="fig/double_banana/")


plot(
    contour(rng, rng, (x,y)->pdf(MvNormal(zeros(2), diagm([3.0, 3.0])), [x,y]), 
    levels=10, c=:vik, label="true",
    colorbar=nothing, showaxis=false, grid=false),
)
# savefig("fig/double_banana/reference.pdf")

plot(
    contour(rng, rng, (x,y)->f([x,y]), levels=10, c=:vik, label="true",
            colorbar=nothing, showaxis=false, grid=false),
)
# savefig("fig/double_banana/target.pdf")

# samples = PSDModels.sample(sar, 5000, threading=true)
# histogram2d([s[1] for s in samples], [s[2] for s in samples], 
#             nbins=100, c=:blues, label="data", normalize=true,
#             xlims=(a,b), ylims=(a,b))


KL_div(a, b) = (1/length(a)) * sum(a .* log.(a ./ b))
Chi2_distance(a, b) = (1/length(a)) * sum((a .- b).^2 ./ (b))
Hell_distance(a,b) = 0.5*(1/length(a)) * sum((a.^0.5 .- b.^0.5).^2 )
pdf_tuple(x) = pdf(sar, [x...])
tar_vec = Iterators.product(rng, rng) |> collect |> x->reshape(x,length(x)) |> x->f.(x)
approx_vec = Iterators.product(rng, rng) |> collect |> x->reshape(x,length(x)) |> x->pdf_tuple.(x)

KL_div(tar_vec, approx_vec)
Chi2_distance(tar_vec, approx_vec)
Hell_distance(tar_vec, approx_vec)

pdf_func1 = PSDModels.pushforward_pdf_function(sar;layers=collect(1:1))
func_wrapper_tuple(func) = x->func([x...])
approx_vec1 = Iterators.product(rng, rng) |> collect |> x->reshape(x,length(x)) |> x->func_wrapper_tuple(pdf_func1).(x)

KL_div(tar_vec, approx_vec1)
Chi2_distance(tar_vec, approx_vec1)
Hell_distance(tar_vec, approx_vec1)


pdf_func2 = PSDModels.pushforward_pdf_function(sar;layers=collect(1:2))
func_wrapper_tuple(func) = x->func([x...])
approx_vec2 = Iterators.product(rng, rng) |> collect |> x->reshape(x,length(x)) |> x->func_wrapper_tuple(pdf_func2).(x)

KL_div(tar_vec, approx_vec2)
Chi2_distance(tar_vec, approx_vec2)
Hell_distance(tar_vec, approx_vec2)