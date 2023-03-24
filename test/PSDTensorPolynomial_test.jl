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