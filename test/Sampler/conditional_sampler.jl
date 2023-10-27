using Test

@testset "PSDModel conditional sampler" begin
    f(x) = sum(x.^2)
    model = PSDModel(Legendre(-1.0..1.0)^2, :downward_closed, 2)
    X = [rand(d) * 2 .- 1 for _=1:500]
    Y = f.(X)
    fit!(model, X, Y, trace=false)
    PSDModels.normalize!(model)
    cond_sampler = ConditionalSampler(model, 1)
    Y = PSDModels.cond_sample(cond_sampler, [[rand() * 2 - 1] for _=1:1000]; threading=false)
    @test all([-1≤xi[1]≤1 for xi in Y])
end

@testset "SelfReinforcedConditionalSampler" begin
    
end