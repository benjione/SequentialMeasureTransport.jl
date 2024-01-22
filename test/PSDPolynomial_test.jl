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
    @testset "1D custom C" begin
        f(x) = x[1]^2
        f_int(x) = 1/3 * x[1]^3
        model = PSDModel(Legendre(), :trivial, 5)
        
        X = collect(range(-1, 1, length=100))
        X = [[x] for x in X]
        Y = f.(X)
        fit!(model, X, Y, maxit=1000, λ_1=1e-5, λ_2=0.0)

        int_model = integral(model, 1; C=0.0)
        @test norm(int_model.(X) .- f_int.(X))/norm(f_int.(X)) < 1e-2
    end

    @testset "1D default C" begin
        f(x) = x[1]^2
        f_int(x) = 1/3 * x[1]^3 + 1/3
        model = PSDModel(Legendre(), :trivial, 5)
        
        X = collect(range(-1, 1, length=100))
        X = [[x] for x in X]
        Y = f.(X)
        fit!(model, X, Y, maxit=1000, λ_1=1e-5, λ_2=0.0)

        int_model = integral(model, 1; C=-1.0)
        int_model2 = integral(model, 1)
        @test norm(int_model.(X) .- f_int.(X))/norm(f_int.(X)) < 1e-2
        @test norm(int_model2.(X) .- f_int.(X))/norm(f_int.(X)) < 1e-2
    end

    @testset "2D custom C" begin
        f(x) = x[1]^2 + x[2]^2
        f_int(x) = (1/3) * x[1]^3 + x[2]^2 * x[1]
        f_int2(x) = x[1]^2 * x[2] + (1/3) * x[2]^3
        model = PSDModel(Legendre()^2, :trivial, 4)
        
        X = [[x,y] for x=collect(range(-1, 1, length=20)), y=collect(range(-1, 1, length=20))]
        X = reshape(X, length(X))
        Y = f.(X)
        fit!(model, X, Y)
        @test norm(model.(X) .- f.(X))/norm(f.(X)) < 1e-2

        int_model = integral(model, 1; C=0.0)
        @test norm(int_model.(X) .- f_int.(X))/norm(f_int.(X)) < 1e-2

        int_model2 = integral(model, 2; C=0.0)
        @test norm(int_model2.(X) .- f_int2.(X))/norm(f_int2.(X)) < 1e-2
    end

    @testset "2D downward_closed custom C" begin
        f(x) = x[1]^2 + x[2]^2
        f_int(x) = (1/3) * x[1]^3 + x[2]^2 * x[1]
        f_int2(x) = x[1]^2 * x[2] + (1/3) * x[2]^3
        model = PSDModel(Legendre()^2, :downward_closed, 2)
        
        X = [[x,y] for x=collect(range(-1, 1, length=20)), y=collect(range(-1, 1, length=20))]
        X = reshape(X, length(X))
        Y = f.(X)
        fit!(model, X, Y)
        @test norm(model.(X) .- f.(X))/norm(f.(X)) < 1e-2

        int_model = integral(model, 1; C=0.0)
        @test norm(int_model.(X) .- f_int.(X))/norm(f_int.(X)) < 1e-2

        int_model2 = integral(model, 2; C=0.0)
        @test norm(int_model2.(X) .- f_int2.(X))/norm(f_int2.(X)) < 1e-2
    end
end

@testset "Density estimation" begin
    @testset "1D" begin
        pdf(x) = exp(-x[1]^2/2)/sqrt(2*pi)
        X = randn(1000)
        X = [[x] for x in X]
        model = PSDModel(Legendre(-15..15), :trivial, 30)
        loss(Z) = -1/length(Z) * sum(log.(Z))
        minimize!(model, loss, X, maxit=5000, normalization_constraint=true)
        domx = collect(range(-15, 15, length=2000))
        @test norm(model.(domx) .- pdf.(domx))/norm(pdf.(domx)) < 1e-1
    end

    @testset "1D downward_closed" begin
        pdf(x) = exp(-x[1]^2/2)/sqrt(2*pi)
        X = randn(1000)
        X = [[x] for x in X]
        model = PSDModel(Legendre(-15..15), :downward_closed, 30)
        loss(Z) = -1/length(Z) * sum(log.(Z))
        minimize!(model, loss, X, maxit=5000, normalization_constraint=true)
        domx = collect(range(-15, 15, length=2000))
        @test norm(model.(domx) .- pdf.(domx))/norm(pdf.(domx)) < 1e-1
    end
end


@testset "Normalization" begin
    @testset "2D" begin
        model = PSDModel(Legendre()^2, :trivial, 4)
        
        model.B[2,2] = 2.0
        normalize_orth_measure!(model)
        @test tr(model.B) ≈ 1.0

        PSDModels.normalize!(model)
        @test tr(model.B) ≈ 1.0
    end

    @testset "2D scaled" begin
        @testset "trivial tensorization" begin
            f(x) = 2*(x[2]-0.5)^2 * (x[2]+0.5)^2 + 2*(x[1]-0.5)^2 * (x[1]+0.5)^2
            model = PSDModel(Legendre(-15..15)^2, :trivial, 20)
            # generate some data
            X = [(rand(2) * 2 .- 1) for i in 1:200]
            Y = f.(X)

            fit!(model, X, Y, maxit=2000)
            normalize_orth_measure!(model)
            @test tr(model.B) ≈ 1.0

            PSDModels.normalize!(model)
            @test tr(model.B) ≈ 1.0
        end
    end
end

@testset "Index Permutation" begin
    @testset "3D downward closed tensorization" begin
        f(x) = 2*(x[2]-0.5)^2 * (x[2]+0.5)^2 + 2*(x[1]-0.5)^2 * (x[1]+0.5)^2
        model = PSDModel(Legendre(-15..15)^2, :downward_closed, 4)
        # generate some data
        X = [(rand(2) * 2 .- 1) for i in 1:200]
        Y = f.(X)

        fit!(model, X, Y, maxit=2000)
        model_perm = PSDModels.permute_indices(model, [2,1])
        for i=1:100
            x = (rand(2) .- 0.5) * 30.0
            @test model_perm(x) == model([x[2], x[1]])
        end
    end
end

@testset "Compile SoS polynomial to polynomial" begin
    @testset "standard compile" begin
        f(x) = 2*(x[2]-0.5)^2 * (x[2]+0.5)^2 + 2*(x[1]-0.5)^2 * (x[1]+0.5)^2
        model = PSDModel(Legendre(-15..15)^2, :downward_closed, 5)
        # generate some data
        X = [(rand(2) * 2 .- 1) for i in 1:200]
        Y = f.(X)

        fit!(model, X, Y, maxit=2000)
        model_poly = PSDModels.compile(model)
        for i=1:100
            x = (rand(2) .- 0.5) * 30.0
            @test model_poly(x) ≈ model(x)
        end
    end

    @testset "integral compile" begin
        f(x) = x[1]^2 + x[2]^2
        f_int(x) = (1/3) * x[1]^3 + x[2]^2 * x[1]
        f_int2(x) = x[1]^2 * x[2] + (1/3) * x[2]^3
        model = PSDModel(Legendre()^2, :downward_closed, 2)
        
        X = [[x,y] for x=collect(range(-1, 1, length=20)), y=collect(range(-1, 1, length=20))]
        X = reshape(X, length(X))
        Y = f.(X)
        fit!(model, X, Y)

        int_model = integral(model, 1; C=0.0)
        int_model_compiled = PSDModels.compiled_integral(model, 1; C=0.0)
        @test all(int_model.(X) .≈ int_model_compiled.(X))

        int_model2 = integral(model, 2; C=0.0)
        int_model_compiled2 = PSDModels.compiled_integral(model, 2; C=0.0)
        @test all(int_model2.(X) .≈ int_model_compiled2.(X))
    end

    @testset "integral compile no C" begin
        f(x) = x[1]^2 + x[2]^2
        f_int(x) = (1/3) * x[1]^3 + x[2]^2 * x[1]
        f_int2(x) = x[1]^2 * x[2] + (1/3) * x[2]^3
        model = PSDModel(Legendre()^2, :downward_closed, 2)
        
        X = [[x,y] for x=collect(range(-1, 1, length=20)), y=collect(range(-1, 1, length=20))]
        X = reshape(X, length(X))
        Y = f.(X)
        fit!(model, X, Y)

        int_model = integral(model, 1)
        int_model_compiled = PSDModels.compiled_integral(model, 1)
        for x in X
            @test isapprox(int_model(x), int_model_compiled(x), atol=1e-10)
        end

        int_model2 = integral(model, 2)
        int_model_compiled2 = PSDModels.compiled_integral(model, 2)
        for x in X
            @test isapprox(int_model2(x), int_model_compiled2(x), atol=1e-10)
        end
    end
end