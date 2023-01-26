using PSDModels
using KernelFunctions
using Test

@testset "Model creation/evaluation" begin
    @testset "simple" begin
        X = Float64[1, 2, 3]
        Y = Float64[1, 1, 1]
        k = MaternKernel()
        model = PSDModel(X, Y, k)
        @test model(1) ≈ 1
        @test model(2) ≈ 1
        @test model(3) ≈ 1

    end

    @testset "Float16 type" begin
        X = Float16[1, 2, 3]
        Y = Float16[1, 1, 1]
        k = MaternKernel()
        model = PSDModel(X, Y, k)
        @test isapprox(model(1), 1, rtol=1e-2)
        @test isapprox(model(3), 1, rtol=1e-2)
        @test isapprox(model(2), 1, rtol=1e-2)
    end
end

@testset "arithmetic" begin 
    @testset "scalar multiplication" begin
        X = Float64[1, 2, 3]
        Y = Float64[1, 1, 1]
        k = MaternKernel()
        model = PSDModel(X, Y, k)

        model2 = 2 * model
        @test model2(1) ≈ 2
        @test model2(2) ≈ 2
        @test model2(3) ≈ 2

        model3 = model * 2
        @test model3(1) ≈ 2
        @test model3(2) ≈ 2
        @test model3(3) ≈ 2
    end
end