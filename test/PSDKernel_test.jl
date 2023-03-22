using KernelFunctions

@testset "Model creation/evaluation" begin
    @testset "simple" begin
        X = Float64[1, 2, 3]
        Y = Float64[1, 1, 1]
        k = MaternKernel()
        model = PSDModel(X, Y, k)
        @test model(1.0) ≈ 1
        @test model(2.0) ≈ 1
        @test model(3.0) ≈ 1

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

    @testset "2D" begin
        X = [rand(2) for i in 1:20]
        Y = Float64[1 for i in 1:20]
        k = MaternKernel()
        model = PSDModel(X, Y, k, solver=:gradient_descent)

        @test isapprox(model([0.5, 0.5]), 1.0, rtol=1e-1)
    end

    @testset "changing X" begin
        X = Float64[1, 2, 3]
        Y = Float64[1, 1, 1]

        k = MaternKernel()
        model = PSDModel(X, Y, k)

        X[1] = 10

        @test model(1.0) ≈ 1
        @test model(2.0) ≈ 1
        @test model(3.0) ≈ 1
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
        @test model2(1.0) ≈ 2
        @test model2(2.0) ≈ 2
        @test model2(3.0) ≈ 2

        model3 = model * 2
        @test model3(1.0) ≈ 2
        @test model3(2.0) ≈ 2
        @test model3(3.0) ≈ 2
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
        model = PSDModel(X, Y, k, 
                        solver=:gradient_descent)

        for x in rand(100)*1.5 .-0.75
            @test isapprox(model(x), f(x), atol=1e-1)
        end
    end

    @testset "function fit!" begin
        f(x) = 2*(x-0.5)^2 * (x+0.5)^2
        N = 20
        X = collect(range(-1, 1, length=N))
        Y = f.(X)

        k = MaternKernel(ν=1.5)
        model = PSDModel(k, X)

        fit!(model, X, Y, maxit=1000)

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

@testset "integrate" begin
    f(x) = 2*(x-0.5)^2 * (x+0.5)^2
    f_int(x) = 0.125*x + 0.4*x^5 - (1/3)*x^3
    N = 30
    X = collect(range(-1, 1, length=N))
    Y = f.(X)

    k = MaternKernel(ν=1.0)
    model = PSDModel(X, Y, k; solver=:gradient_descent)

    for x in rand(100) .- 0.5
        @test isapprox(integrate(model, 0..x), f_int(x), atol=1e-1)
    end
end

@testset "Density estimation" begin
    X = randn(300) * 0.75 .+ 0.5
    pdf_X(x) = 1/(sqrt(2*pi*0.75)) * exp(-(x-0.5)^2/(2*0.75))

    S = randn(100) * 0.75 .+ 0.5
    k = MaternKernel(ν=1.5)
    model = PSDModel(k, S)

    loss(Z) = -1/length(Z) * mapreduce(i->log(Z[i]), +, 1:length(Z))

    # TODO rewrite once the constraint minimization is done as density estimation
    minimize!(model, loss, X, maxit=1000)

    model = (1/integrate(model, -5..5, amount_quadrature_points=100)) * model

    dom_x = collect(range(-2, 3, length=200))

    # Maybe more meaningful test with expected error bound from literature?
    @test norm(pdf_X.(dom_x) - model.(dom_x)) < 0.33
end

@testset "add support" begin
    X = Float64[1, 2, 3]
    Y = Float64[1, 1, 1]
    k = MaternKernel()
    model = PSDModel(X, Y, k)

    X2 = Float64[4, 5, 6]
    model2 = PSDModels.add_support(model, X2)

    for i in rand(20) * 4
        @test model(i) ≈ model2(i)
    end
end