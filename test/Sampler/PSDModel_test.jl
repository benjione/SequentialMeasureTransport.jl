

@testset "Marginal pdf = pdf for dC = 0" begin
    f(x) = sum(x[1]^2) + sum(x[2]^4)
    model = PSDModel(Legendre(-1.0..1.0)^2, :downward_closed, 1)
    X = [rand(2) * 2 .- 1 for _=1:500]
    Y = f.(X)
    fit!(model, X, Y, trace=false)
    smp = Sampler(model)
    
    for _=1:10
        x = SMT.sample(smp)
        @test pdf(smp, x) == SMT.marginal_pdf(smp, x)
    end
end