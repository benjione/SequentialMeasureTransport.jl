


@testset "Building large sampler" begin
    for d=2:10
        @testset "$(d) dimension" begin
            f(x) = sum(x.^2)
            model = PSDModel(Legendre(-1..1)^d, :downward_closed, 1)
            X = [rand(d) * 2 .- 1 for _=1:500]
            Y = f.(X)
            fit!(model, X, Y, trace=false)
            sampler = Sampler(model)
            x = PSDModels.sample(sampler)
            @test all([-1≤xi≤1 for xi in x])
        end
    end
end