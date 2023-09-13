using KernelFunctions: MaternKernel

@testset "Model creation/evaluation" begin
    @testset "simple" begin
        X = Float64[1, 2, 3]
        X = [[x] for x in X]
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
        X = [[x] for x in X]
        Y = Float64[1, 1, 1]

        k = MaternKernel()
        model = PSDModel(X, Y, k)

        X[1][1] = 10.0

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
        X = [[x] for x in X]
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
        N = 25
        X = collect(range(-1, 1, length=N))
        Y = f.(X)
        X = [[x] for x in X]

        k = MaternKernel(ν=1.0)
        model = PSDModel(X, Y, k; solver=:direct)

        domx = collect(range(-1, 1, length=1000))
        @test norm(model.(domx) - f.(domx))/norm(f.(domx)) < 3e-1
    end


    @testset "gradient descent fit" begin
        f(x) = 2*(x-0.5)^2 * (x+0.5)^2
        N = 30
        X = collect(range(-1, 1, length=N))
        Y = f.(X)
        X = [[x] for x in X]

        k = MaternKernel(ν=1.0)
        model = PSDModel(X, Y, k, 
                        solver=:gradient_descent)

        domx = collect(range(-1, 1, length=1000))
        @test norm(model.(domx) - f.(domx))/norm(f.(domx)) < 1e-1
    end

    @testset "function fit!" begin
        f(x) = 2*(x-0.5)^2 * (x+0.5)^2
        N = 30
        X = collect(range(-1, 1, length=N))
        Y = f.(X)
        X = [[x] for x in X]

        k = MaternKernel(ν=1.0)
        model = PSDModel(k, X)

        fit!(model, X, Y, λ_1=1e-4, λ_2=1e-8, maxit=2000)

        domx = collect(range(-1, 1, length=1000))
        @test norm(model.(domx) - f.(domx))/norm(f.(domx)) < 1.5e-1

    end
end

@testset "gradient" begin
    f(x) = 2*(x-0.5)^2 * (x+0.5)^2
    ∇f(x) = 4*(x-0.5)*(x+0.5)^2 + 4*(x-0.5)^2*(x+0.5)
    N = 30
    X = collect(range(-1, 1, length=N))
    Y = f.(X)
    X = [[x] for x in X]

    k = MaternKernel(ν=1.0)
    model = PSDModel(X, Y, k; solver=:gradient_descent)

    for x in rand(100).- 0.5
        @test isapprox(gradient(model, [x])[1], ∇f(x), atol=3e-1, rtol=2e-1)
    end
end

@testset "integrate" begin
    f(x) = 2*(x-0.5)^2 * (x+0.5)^2
    f_int(x) = 0.125*x[1] + 0.4*x[1]^5 - (1/3)*x[1]^3
    N = 30
    X = collect(range(-1, 1, length=N))
    Y = f.(X)
    X = [[x] for x in X]

    k = MaternKernel(ν=1.0)
    model = PSDModel(X, Y, k; solver=:gradient_descent)

    @inline interval(x) = 0..x[1]
    int_vec = PSDModels.integrate.(Ref(model), interval.(X))
    @test norm(int_vec - f_int.(X)) < 1e-1
end

## has probabilistic behavior in testing, skip for now
# @testset "Density estimation" begin
#     X = randn(300) * 0.75 .+ 0.5
#     pdf_X(x) = 1/(sqrt(2*pi*0.75)) * exp(-(x-0.5)^2/(2*0.75))

#     S = randn(120) * 0.75 .+ 0.5
#     k = MaternKernel(ν=1.5)
#     model = PSDModel(k, S)

#     loss(Z) = -1/length(Z) * mapreduce(i->log(Z[i]), +, 1:length(Z))

#     # TODO rewrite once the constraint minimization is done as density estimation
#     minimize!(model, loss, X, maxit=1500, trace=false)

#     model = (1/integrate(model, -5..5, amount_quadrature_points=100)) * model

#     dom_x = collect(range(-2, 3, length=200))

#     # Maybe more meaningful test with expected error bound from literature?
#     @test norm(pdf_X.(dom_x) - model.(dom_x)) < 0.33
# end

@testset "add support" begin
    X = Float64[1, 2, 3]
    X = [[x] for x in X]
    Y = Float64[1, 1, 1]
    k = MaternKernel()
    model = PSDModel(X, Y, k)

    X2 = Float64[4, 5, 6]
    X2 = [[x] for x in X2]
    model2 = PSDModels.add_support(model, X2)

    for i in rand(20) * 4
        @test model(i) ≈ model2(i)
    end
end