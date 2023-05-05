using PSDModels 
using LinearAlgebra
using ApproxFun
using Plots

# dimension of model
d = 2

f(x) = norm(x)^2
model = PSDModel(Legendre(-1..1)^d, :downward_closed, 1)

X = [rand(d) * 2 .- 1 for _=1:1000]
Y = f.(X)
fit!(model, X, Y, trace=true)

sampler = Sampler(model)

# N_sample = 1000
# X = zeros(N_sample, d)
# sample_in_sphere(sampler) = begin
#     x = PSDModels.sample(sampler)
#     while norm(x)>1.0
#         x = PSDModels.sample(sampler)
#     end
#     return x
# end
# @time Threads.@threads for i=1:N_sample
#     X[i,:] = sample_in_sphere(sampler)
# end

# set Q(x) ≤ 1, Q(x) = x_1^2 + x_2^2 + ...
function boundary_derivative(x)
    return [2 * x_i for x_i in x]
end
# function to generate poin on the d dimensional sphere from the angles
on_d_sphere(d, angles) = [[prod(sin, angles[1:i-1], init=1.0) * cos(angles[i]) for i=1:d-1];
                            prod(sin, angles[1:d-1])]

N_amount_angles = 100
@info "sampling on the $(d) dimensional sphere, with $(N_amount_angles^(d-1)) points used"
angle_ranges = StepRangeLen[range(0, π, length=N_amount_angles) for _=1:d-2]
# angle_range_last = range(0, 2π, length=N_amount_angles)
push!(angle_ranges, range(0, 2π, length=N_amount_angles))

points_on_sphere = [on_d_sphere(d, angles) for angles in Iterators.product(angle_ranges...)]
points_on_sphere = reshape(points_on_sphere, length(points_on_sphere))
N_sample = 1000

points_rectangle = [
    [[0.5, x] for x in range(-0.5, 0.5, length=100)];
    [[-0.5, x] for x in range(-0.5, 0.5, length=100)];
    [[x, 0.5] for x in range(-0.5, 0.5, length=100)];
    [[x, -0.5] for x in range(-0.5, 0.5, length=100)]
]

if d==2
    points_pullbacked = PSDModels.pullback_x.(Ref(sampler), points_on_sphere)
    plot(
        scatter([i[1] for i in points_on_sphere], 
        [i[2] for i in points_on_sphere], legend=nothing,
        xlabel="x", ylabel="y"),
        scatter([i[1] for i in points_pullbacked], 
        [i[2] for i in points_pullbacked], legend=nothing)
    )
end

X = @time PSDModels.sample_subdomain(sampler, points_on_sphere, 
                boundary_derivative, N_sample, threading=true)
X = PSDModels.unslice_matrix(X)
X = X'
if d==2
    domx = range(-1, 1, length=100)
    domy = range(-1, 1, length=100)
    plot(
        scatter(X[:,1], X[:,2], legend=nothing),
        contourf(domx, domy, (x,y)->model([x,y]),levels=5,
                color=:turbo, 
                clabels=false, 
                cbar=true, lw=1),
    )
end

rand_dom = [rand(d) *2 .- 1 for _=1:N_sample]
filter!(x->norm(x)≤1.0, rand_dom)
sort!(rand_dom, by=x->norm(x))
X_ax = map(x->norm(x), eachrow(X))
filter!(x->x≤1.0, X_ax)
domx = range(0.0, 1.0, length=100)
histogram(X_ax, normed=true, label="samples", alpha=0.2, bins=100,
    xlabel="\$R\$", ylabel="density", color=:blue)
plot!(domx, domx.^(1+d)*(2+d), label="expected", color=:auto, lw=3)
title!("dimension \$d=$(d)\$")
savefig("histogram_d$(d).pdf")
# plot!(norm.(rand_dom), model.(rand_dom)*3, label="model", lw=3, 
#         color=:auto, linestyle=:dash)
# scatter!([d/3], [0.0], label="expected mean")
# scatter!([(1/length(rand_dom)) * sum(X_ax)], [0.0], label="mean")
