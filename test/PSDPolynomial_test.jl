using ApproxFun

@testset "Model fit" begin
    @testset "1D fit" begin

        f(x) = 2*(x-0.5)^2 * (x+0.5)^2
        model = PSDModel(Chebyshev(), 10)

        X = rand(200) * 2 .-1
        Y = f.(X)

        fit!(model, X, Y, trace=false)
        
        test_X = range(-1, 1, length=100)
        @test norm(model.(test_X) .- f.(test_X), Inf) < 1e-1
    end

    @testset "2D fit" begin
        
        f(x) = 2*(x[2]-0.5)^2 * (x[2]+0.5)^2 +
               2*(x[1]-0.5)^2 * (x[1]+0.5)^2
        model = PSDModel(Chebyshev()^2, 15)

        X = [(rand(2) * 2 .-1) for _=1:200]
        Y = f.(X)

        fit!(model, X, Y, trace=false)
        
        test_X = [[x,y] for x in range(-1,1,100), y in range(-1,1,100)]
        @test norm(model.(test_X) .- f.(test_X), Inf) < 1e-1
    end
end
