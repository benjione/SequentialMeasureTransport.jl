using PSDModels
using KernelFunctions
using DomainSets
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

    # @testset "Float16 type" begin
    #     X = Float16[1, 2, 3]
    #     Y = Float16[1, 1, 1]
    #     k = MaternKernel()
    #     model = PSDModel(X, Y, k)
    #     @test isapprox(model(1), 1, rtol=1e-2)
    #     @test isapprox(model(3), 1, rtol=1e-2)
    #     @test isapprox(model(2), 1, rtol=1e-2)
    # end

    @testset "changing X" begin
        X = Float64[]
        Y = Float64[1, 1, 1]

        append!(X, Float64[1, 2, 3])

        k = MaternKernel()
        model = PSDModel(X, Y, k)

        append!(X, rand(10))
        X[1] = 10

        @test model(1) ≈ 1
        @test model(2) ≈ 1
        @test model(3) ≈ 1
    end

    # @testset "wiew with resizing X" begin
    #     X = Float64[]
    #     Y = Float64[1, 1, 1]

    #     append!(X, Float64[1, 2, 3])
    #     k = MaternKernel()
    #     model = PSDModel(X, Y, k, use_view=true)

    #     append!(X, rand(10))

    #     @test model(1) ≈ 1
    #     @test model(2) ≈ 1
    #     @test model(3) ≈ 1
    # end
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

@testset "Model fitting" begin
    @testset "direct fit" begin
        f(x) = 2*(x-0.5)^2 * (x+0.5)^2
        N = 20
        X = collect(range(-1, 1, length=N))
        Y = f.(X)

        k = MaternKernel(ν=1.0)
        model = PSDModel(X, Y, k; solver=:direct)

        for x in rand(100)*1.5 .-0.75
            @test isapprox(model(x), f(x), atol=1e-1)
        end
    end


    @testset "gradient descent fit" begin
        f(x) = 2*(x-0.5)^2 * (x+0.5)^2
        N = 20
        X = collect(range(-1, 1, length=N))
        Y = f.(X)

        k = MaternKernel(ν=1.0)
        model = PSDModel(X, Y, k; solver=:gradient_descent)

        for x in rand(100)*1.5 .-0.75
            @test isapprox(model(x), f(x), atol=1e-1)
        end
    end
end

@testset "gradient" begin
    f(x) = 2*(x-0.5)^2 * (x+0.5)^2
    ∇f(x) = 4*(x-0.5)*(x+0.5)^2 + 4*(x-0.5)^2*(x+0.5)
    N = 30
    X = collect(range(-1, 1, length=N))
    Y = f.(X)

    k = MaternKernel(ν=1.0)
    model = PSDModel(X, Y, k; solver=:gradient_descent)

    for x in rand(100).- 0.5
        @test isapprox(gradient(model, x), ∇f(x), atol=3e-1, rtol=2e-1)
    end
end

@testset "integral" begin
    f(x) = 2*(x-0.5)^2 * (x+0.5)^2
    f_int(x) = 0.125*x + 0.4*x^5 - (1/3)*x^3
    N = 30
    X = collect(range(-1, 1, length=N))
    Y = f.(X)

    k = MaternKernel(ν=1.0)
    model = PSDModel(X, Y, k; solver=:gradient_descent)

    for x in rand(100) .- 0.5
        @test isapprox(integral(model, 0..x), f_int(x), atol=1e-1)
    end
end