using ApproxFun

@testset "Marginalization" begin
    @testset "orthogonal 2D" begin
        model = PSDModel(Legendre()^2, :trivial, 4)
        
        model.B[2,2] = 2.0
        @test model.B == diagm([1.0, 2.0, 1.0, 1.0])
        model_marginalized = marginalize_orth_measure(model, 1)
        @test model_marginalized.B == diagm([3.0, 2.0])
        model_marginalized2 = marginalize(model, 1)
        @test model_marginalized2.B ≈ model_marginalized.B
    end
end

@testset "Integration" begin
    @testset "simple 1D" begin
        f(x) = x^2
        f_int(x) = 1/3 * x^3
        model = PSDModel(Legendre(), :trivial, 4)
        
        X = collect(range(-1, 1, length=100))
        Y = f.(X)
        fit!(model, X, Y, maxit=500)

        int_model = integral(model, 1)
        @test norm(int_model.(X) .- f_int.(X))/norm(f_int.(X)) < 1e-2
    end

    @testset "2D" begin
        f(x) = x[1]^2 + x[2]^2
        f_int(x) = (1/3) * x[1]^3 + x[2]^2 * x[1]
        f_int2(x) = x[1]^2 * x[2] + (1/3) * x[2]^3
        model = PSDModel(Legendre()^2, :trivial, 4)
        
        X = [[x,y] for x=collect(range(-1, 1, length=20)), y=collect(range(-1, 1, length=20))]
        X = reshape(X, length(X))
        Y = f.(X)
        fit!(model, X, Y)
        @test norm(model.(X) .- f.(X))/norm(f.(X)) < 1e-2

        int_model = integral(model, 1)
        @test norm(int_model.(X) .- f_int.(X))/norm(f_int.(X)) < 1e-2

        int_model2 = integral(model, 2)
        @test norm(int_model2.(X) .- f_int2.(X))/norm(f_int2.(X)) < 1e-2
    end
end