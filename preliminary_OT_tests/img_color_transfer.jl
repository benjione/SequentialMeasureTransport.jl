using Images
import SequentialMeasureTransport as SMT
using LinearAlgebra
using Transducers
using Plots
using StaticArrays

include("helpers.jl")


img1 = load("preliminary_OT_tests/data/pink_bananas.jpg")
img2 = load("preliminary_OT_tests/data/Vancouver.jpg")

# img1 = imresize(img1, ratio=0.25)

length1 = length(img1)
length2 = length(img2)

target_size = 30000
iter_1_count = length1 ÷ target_size
iter_2_count = length2 ÷ target_size

colors_img1 = [[Float64(ColorTypes.red(img1[i])),
                Float64(ColorTypes.green(img1[i])),
                Float64(ColorTypes.blue(img1[i]))] for i in 1:iter_1_count:length1]

colors_img2 = [[Float64(ColorTypes.red(img2[i])),
                Float64(ColorTypes.green(img2[i])),
                Float64(ColorTypes.blue(img2[i]))] for i in 1:iter_2_count:length2]


gaussian(x::AbstractVector{T}, y::AbstractVector{T}; σ=0.05) where {T<:Number} = (1)/(2π * σ^2)^(3/2) * exp(-norm(x - y)^2 / (2 * σ^2))

function density_map(x::AbstractVector{T}, color_map::AbstractVector{<:Vector{T}}) where {T}
    (1/length(color_map)) * foldxl(+, color_map |> Map(y -> gaussian(x, y)); init=zero(T), simd=true)
end


@time density_map(rand(3), colors_img1)
@time density_map(rand(3), colors_img2)

size_x = 15
rng = range(0, 1, length=size_x)
left_marg_vec = zeros([length(rng) for _ in 1:3]...)
@time Threads.@threads for i=1:size_x
    x = rng[i]
    for (j, y) in enumerate(rng)
        for (k, z) in enumerate(rng)
            @inbounds left_marg_vec[i, j, k] = density_map(SA[x, y, z], colors_img1)
        end
    end
end
## normalize Vector
left_marg_vec .*= (size_x^3 / sum(left_marg_vec))
plot(1:size_x, i->sum(left_marg_vec[i, :, :]), color=:red, linewidth=2, label="red")
plot!(1:size_x, i->sum(left_marg_vec[:, i, :]), color=:green, linewidth=2, label="green")
plot!(1:size_x, i->sum(left_marg_vec[:, :, i]), color=:blue, linewidth=2, label="blue")

right_marg_vec = zeros([length(rng) for _ in 1:3]...)
@time Threads.@threads for i=1:size_x
    x = rng[i]
    for (j, y) in enumerate(rng)
        for (k, z) in enumerate(rng)
            @inbounds right_marg_vec[i, j, k] = density_map(SA[x, y, z], colors_img2)
        end
    end
end
## normalize Vector
right_marg_vec .*= (size_x^3 / sum(right_marg_vec))
plot(1:size_x, i->sum(right_marg_vec[i, :, :]), color=:red, linewidth=2, label="red")
plot!(1:size_x, i->sum(right_marg_vec[:, i, :]), color=:green, linewidth=2, label="green")
plot!(1:size_x, i->sum(right_marg_vec[:, :, i]), color=:blue, linewidth=2, label="blue")


## Compute OT
rng_Sink = Iterators.product(rng, rng, rng)
c(x, y) = norm(x - y)^2
@time M_sink = compute_Sinkhorn(rng_Sink, vec(left_marg_vec), 
            vec(right_marg_vec), c, 0.01; iter=100)

# contour(M_sink')

rng_map_vec = rng_Sink |> Map(x -> SA[x...]) |> collect


cart_to_linear = reshape(collect(1:length(left_marg_vec)), size_x, size_x, size_x)
function color_to_index(color; size_x=size_x)
    x_1 = round(Int, color[1] * (size_x-1) + 1)
    x_2 = round(Int, color[2] * (size_x-1) + 1)
    x_3 = round(Int, color[3] * (size_x-1) + 1)
    return cart_to_linear[x_1, x_2, x_3]
end

function RGB_to_SA(rgb)
    return SA[ColorTypes.red(rgb) |> Float64, 
                ColorTypes.green(rgb) |> Float64, 
                ColorTypes.blue(rgb) |> Float64]
end


function color_transfer(img::AbstractArray{T, 2}, M_sink, marginal) where {T}
    img_size = size(img)
    new_img = similar(img)
    Threads.@threads for i=1:img_size[1]
        for j=1:img_size[2]
            color_vec = RGB_to_SA(img[i, j])
            color_index = color_to_index(color_vec)
            ret = Barycentric_map_from_sinkhorn(M_sink, vec(marginal), rng_map_vec, color_index)
            ret = min.(ret, 1.0)
            new_img[i, j] = RGB(ret...)
        end
    end
    return new_img
end

img1
img2
new_img2 = color_transfer(img2, M_sink', right_marg_vec)
new_img1 = color_transfer(img1, M_sink, left_marg_vec)


img1
img2
new_img1
new_img2

## check that color spectrum is correct afterwards

colors_new_img1 = [[Float64(ColorTypes.red(new_img1[i])),
                Float64(ColorTypes.green(new_img1[i])),
                Float64(ColorTypes.blue(new_img1[i]))] for i in 1:iter_1_count:length1]

colors_new_img2 = [[Float64(ColorTypes.red(new_img2[i])),
                Float64(ColorTypes.green(new_img2[i])),
                Float64(ColorTypes.blue(new_img2[i]))] for i in 1:iter_2_count:length2]


rng = range(0, 1, length=size_x)
color_density_new1 = zeros([length(rng) for _ in 1:3]...)
@time Threads.@threads for i=1:size_x
    x = rng[i]
    for (j, y) in enumerate(rng)
        for (k, z) in enumerate(rng)
            @inbounds color_density_new1[i, j, k] = density_map(SA[x, y, z], colors_new_img1)
        end
    end
end
## normalize Vector
color_density_new1 .*= (size_x^3 / sum(color_density_new1))
plot(1:size_x, i->sum(color_density_new1[i, :, :]), color=:red, linewidth=2, label="red")
plot!(1:size_x, i->sum(color_density_new1[:, i, :]), color=:green, linewidth=2, label="green")
plot!(1:size_x, i->sum(color_density_new1[:, :, i]), color=:blue, linewidth=2, label="blue")
## compare to what it should be
plot!(1:size_x, i->sum(right_marg_vec[i, :, :]), color=:red, linewidth=2, linestyle=:dot, label="reference red")
plot!(1:size_x, i->sum(right_marg_vec[:, i, :]), color=:green, linewidth=2, linestyle=:dot, label="reference green")
plot!(1:size_x, i->sum(right_marg_vec[:, :, i]), color=:blue, linewidth=2, linestyle=:dot, label="reference blue")


color_density_new2 = zeros([length(rng) for _ in 1:3]...)
@time Threads.@threads for i=1:size_x
    x = rng[i]
    for (j, y) in enumerate(rng)
        for (k, z) in enumerate(rng)
            @inbounds color_density_new2[i, j, k] = density_map(SA[x, y, z], colors_new_img2)
        end
    end
end
## normalize Vector
color_density_new2 .*= (size_x^3 / sum(color_density_new2))
plot(1:size_x, i->sum(color_density_new2[i, :, :]), color=:red, linewidth=2, label="red")
plot!(1:size_x, i->sum(color_density_new2[:, i, :]), color=:green, linewidth=2, label="green")
plot!(1:size_x, i->sum(color_density_new2[:, :, i]), color=:blue, linewidth=2, label="blue")
## compare to what it should be
plot!(1:size_x, i->sum(left_marg_vec[i, :, :]), color=:red, linewidth=2, linestyle=:dot, label="reference red")
plot!(1:size_x, i->sum(left_marg_vec[:, i, :]), color=:green, linewidth=2, linestyle=:dot, label="reference green")
plot!(1:size_x, i->sum(left_marg_vec[:, :, i]), color=:blue, linewidth=2, linestyle=:dot, label="reference blue")

# save("preliminary_OT_tests/data/new_img2.jpg", new_img2)