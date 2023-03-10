using ApproxFun

@testset "Model fit" begin
    @testset "custom feature map" begin

        Φ(x) = Float64[1, sin(x), cos(x)]
        X = Float64[1, 2, 3]
        Y = Float64[1, 1, 1]
        model = PSDModel(Φ, 3)
        fit!(model, X, Y, trace=false)
        @test isapprox(model(1.0),1.0,atol=1e-1)
        @test isapprox(model(2.0),1.0,atol=1e-1)
        @test isapprox(model(3.0),1.0,atol=1e-1)
    end

    @testset "custom Chebyshev feature map" begin
        f_list = [Fun(Chebyshev(0..3), Float64[zeros(d); 1.0]) for d=0:10]
        Φ(x) = map(f-> f(x), f_list)
        X = Float64[1, 2, 3]
        Y = Float64[1, 1, 1]
        model = PSDModel(Φ, length(f_list))
        fit!(model, X, Y, trace=true)
        @test isapprox(model(1.0),1.0,atol=1e-1)
        @test isapprox(model(2.0),1.0,atol=1e-1)
        @test isapprox(model(3.0),1.0,atol=1e-1)
    end

end


@testset "arithmetic" begin
    @testset "scalar multiplication" begin

        Φ(x) = Float64[1, sin(x), cos(x)]
        X = Float64[1, 2, 3]
        Y = Float64[1, 1, 1]
        model = PSDModel(Φ, 3)
        fit!(model, X, Y, trace=false)
        model = 2 * model
        @test isapprox(model(1.0),2.0,atol=1e-1)
        @test isapprox(model(2.0),2.0,atol=1e-1)
        @test isapprox(model(3.0),2.0,atol=1e-1)
    end
end