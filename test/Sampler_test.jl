using ApproxFun


@testset "Building large sampler" begin
    for d=2:10
        @testset "$(d) dimension" begin
            f(x) = sum(x.^2)
            model = PSDModel(Legendre(-1..1)^d, :downward_closed, 1)
            X = [rand(d) * 2 .- 1 for _=1:500]
            Y = f.(X)
            fit!(model, X, Y, trace=false)
            sampler = Sampler(model)
            x = PSDModels.sample(sampler)
            @test all([-1≤xi≤1 for xi in x])
        end
    end
end

@testset "Reference map and reference sampling" begin
    @testset "Scaling Reference" begin
        f(x) = sum(x.^2)
        model = PSDModel(Legendre(-5.0..(-3.0)) ⊗ Legendre(3.0..10.0), :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            1, :Chi2;
            N_sample=500,
        )
        x = PSDModels.sample_reference(sra)
        @test -5.0 ≤ x[1] ≤ -3.0
        @test 3.0 ≤ x[2] ≤ 10.0

        N = 20
        X = PSDModels.sample_reference(sra, N)
        @test length(X) == N
        @test length(X[1]) == 2
        for X_i in X
            @test -5.0 ≤ X_i[1] ≤ -3.0
            @test 3.0 ≤ X_i[2] ≤ 10.0
        end
    end

    @testset "Gaussian Reference" begin
        f(x) = exp(-sum(x.^2))
        model = PSDModel(Legendre()^2, 
                        :downward_closed, 1; mapping=:algebraicOMF)
        sra = SelfReinforcedSampler(
            f,
            model,
            1, :Chi2;
            N_sample=500,
        )

        N = 100
        X = PSDModels.sample_reference(sra, N)
        @test length(X) == N
        @test length(X[1]) == 2
        @test abs.((1/N) * sum(X)) ≤ [0.5, 0.5] # Check that the mean is close to zero  
    end
end


@testset "self SelfReinforcedSampler" begin
    @testset "simple" begin
        f(x) = sum(x.^2 + x.^4)
        model = PSDModel(Legendre(-1.0..1.0)^2, :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2;
            N_sample=500,
        )
        for _=1:10
            x = PSDModels.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
    end

    @testset "OMF" begin
        f(x) = exp(-0.1*sum(x.^2))
        model = PSDModel(Legendre()^2, 
                    :downward_closed, 3, mapping=:algebraicOMF,
                    λ_1=0.0, λ_2=0.0)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2;
            relaxation_method=:algebraic,
            N_sample=1000,
            reference_map=PSDModels.GaussianReference{2, Float64}(2.0),
        )
        x = PSDModels.sample(sra)
    end

    @testset "broadcasted target" begin
        f(X) = map(x->exp(-sum(x.^2)), X)
        model = PSDModel(Legendre()^2, :downward_closed, 3)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2;
            N_sample=500,
            broadcasted_tar_pdf=true,
            trace=false,
        )
        for _=1:10
            x = PSDModels.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
    end

    @testset "broadcasted target OMF" begin
        f(X) = map(x->exp(-sum(x.^2)), X)
        model = PSDModel(Legendre()^2, :downward_closed, 2,
                    mapping=:algebraicOMF)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2;
            N_sample=500,
            broadcasted_tar_pdf=true,
            trace=false,
        )
        x = PSDModels.sample(sra)
    end

    @testset "Sampling irregular domain" begin
        f(x) = sin(sum(x))
        model = PSDModel(Legendre(-1.0..0.0) ⊗ Legendre(1.0..2.0), 
                    :downward_closed, 3)
        sra = SelfReinforcedSampler(
            f,
            model,
            1, :Chi2;
            relaxation_method=:algebraic,
            N_sample=1000,
            trace=false,
            λ_1=1e-3
        )
        x = PSDModels.sample(sra)
    end

    @testset "broadcasted irregular domain" begin
        f(X) = map(x->sin(sum(x)), X)
        model = PSDModel(Legendre(-1.0..0.0) ⊗ Legendre(1.0..2.0), :downward_closed, 2)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2;
            N_sample=500,
            broadcasted_tar_pdf=true,
            trace=false,
        )
        x = PSDModels.sample(sra)
    end
end