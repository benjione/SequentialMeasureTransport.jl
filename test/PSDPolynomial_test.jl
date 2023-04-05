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

    @testset "orthogonal 2D scaled" begin
        @testset "trivial tensorization" begin
            f(x) = 2*(x[2]-0.5)^2 * (x[2]+0.5)^2 + 2*(x[1]-0.5)^2 * (x[1]+0.5)^2
            model = PSDModel(Legendre(-15..15)^2, :trivial, 20)
            # generate some data
            X = [(rand(2) * 2 .- 1) for i in 1:200]
            Y = f.(X)

            fit!(model, X, Y, maxit=2000)
            model_marginalized = marginalize_orth_measure(model, 1)
            model_marginalized2 = marginalize(model, 1)
            @test model_marginalized2.B ≈ model_marginalized.B

            @test marginalize(model_marginalized, 1) ≈ marginalize(model_marginalized2, 1)
        end

        @testset "downward closed tensorization" begin
            f(x) = 2*(x[2]-0.5)^2 * (x[2]+0.5)^2 + 2*(x[1]-0.5)^2 * (x[1]+0.5)^2
            model = PSDModel(Legendre(-15..15)^2, :downward_closed, 5)
            # generate some data
            X = [(rand(2) * 2 .- 1) for i in 1:200]
            Y = f.(X)

            fit!(model, X, Y, maxit=2000)
            model_marginalized = marginalize_orth_measure(model, 1)
            model_marginalized2 = marginalize(model, 1)
            @test model_marginalized2.B ≈ model_marginalized.B

            @test marginalize(model_marginalized, 1) ≈ marginalize(model_marginalized2, 1)
        end
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

    @testset "2D downward_closed" begin
        f(x) = x[1]^2 + x[2]^2
        f_int(x) = (1/3) * x[1]^3 + x[2]^2 * x[1]
        f_int2(x) = x[1]^2 * x[2] + (1/3) * x[2]^3
        model = PSDModel(Legendre()^2, :downward_closed, 2)
        
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

@testset "Density estimation" begin
    @testset "1D" begin
        pdf(x) = exp(-x^2/2)/sqrt(2*pi)
        X = randn(500)
        model = PSDModel(Legendre(-15..15), :trivial, 30)
        loss(Z) = -1/length(Z) * sum(log.(Z))
        minimize!(model, loss, X, maxit=3000, normalization_constraint=true)
        domx = collect(range(-15, 15, length=2000))
        @test norm(model.(domx) .- pdf.(domx))/norm(pdf.(domx)) < 1e-1
    end

    @testset "1D downward_closed" begin
        pdf(x) = exp(-x^2/2)/sqrt(2*pi)
        X = randn(500)
        model = PSDModel(Legendre(-15..15), :downward_closed, 30)
        loss(Z) = -1/length(Z) * sum(log.(Z))
        minimize!(model, loss, X, maxit=3000, normalization_constraint=true)
        domx = collect(range(-15, 15, length=2000))
        @test norm(model.(domx) .- pdf.(domx))/norm(pdf.(domx)) < 1e-1
    end
end