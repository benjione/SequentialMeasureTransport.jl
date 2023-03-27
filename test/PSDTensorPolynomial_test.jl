using ApproxFun

@testset "Marginalization" begin
    @testset "simple 2D" begin
        f(x) = 1 + x[1]^2 + x[2]^2 + x[1]^2*x[2]^2
        f_marg1(x) = 3 + x^2
        model = PSDModel(Legendre()^2, :trivial, 4)
        
        model.B[2,2] = 2.0
        @test model.B == diagm([1.0, 2.0, 1.0, 1.0])
        model_marginalized = marginalize_orth_measure(model, 1)
        @test model_marginalized.B == diagm([3.0, 2.0])
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
end