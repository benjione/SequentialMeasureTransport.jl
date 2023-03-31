using PSDModels
using BenchmarkTools
using ApproxFun

model = PSDModel(Chebyshev()^2, :trivial, 50)

# warmup
model([0.5, 0.5])

N = 100
time_vec = Float64[]
for i in 1:5:400
    model = PSDModel(Chebyshev()^2, :trivial, i)
    time = 0.0
    for _ = 1:N
        time += @elapsed model([0.5, 0.5])
    end 
    push!(time_vec, time/N)
end

using Plots
plot(1:5:200, time_vec, label="PSDModels")
xlabel!("dimension feature map")
ylabel!("time [ms]")