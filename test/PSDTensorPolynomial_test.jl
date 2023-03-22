using ApproxFun

@testset "Marginalization" begin
    @testset "simple 2D" begin
        f(x) = 1 + x[1]^2 + x[2]^2 + x[1]^2*x[2]^2
        f_marg(x) = 3 + x^2
        model = PSDModel(Legendre()^2, :trivial, 4)
        
        model.B[2,2] = 2.0
        @test model.B == diagm([1.0, 2.0, 1.0, 1.0])
        model_marginalized = marginalize_orth_measure(model, 1)
        @test model_marginalized.B == diagm([3.0, 2.0])
    end

    @testset "2D with fit" begin
        f(x) = 1 + x[1]^2 + x[2]^2 + x[1]^2*x[2]^2
        f_marg(x) = 3 + x^2
        model = PSDModel(Legendre()^2, :trivial, 4)

        N = 20
        dom_x = range(-1, 1, length=N)
        dom_y = range(-1, 1, length=N)
        dom_xy = [[x,y] for (x,y) in zip(range(-1,1,N),range(-1,1,N))]
        fit!(model, dom_xy, f.(dom_xy), maxit=1000)
        @test norm(model.(dom_xy) .- f.(dom_xy))/norm(f.(dom_xy)) < 1e-1
        @test norm(model_marginalized.(dom_x) .- f_marg.(dom_x))/norm(f_marg.(dom_x)) < 1e-1

    end
end