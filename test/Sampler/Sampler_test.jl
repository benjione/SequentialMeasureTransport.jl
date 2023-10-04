using ApproxFun


@testset "Building large sampler" begin
    for d=2:10
        @testset "$(d) dimension" begin
            f(x) = sum(x.^2)
            model = PSDModel(Legendre(-1.0..1.0)^d, :downward_closed, 1)
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
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            1, :Chi2U,
            PSDModels.ScalingReference{2}([-5.0, 3.0], [-3.0, 10.0]);
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
        model = PSDModel(Legendre(0.0..1.0)^2, 
                        :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            1, :Chi2U,
            PSDModels.GaussianReference{2, Float64}(1.0);
            N_sample=500,
        )

        N = 100
        X = PSDModels.sample_reference(sra, N)
        @test length(X) == N
        @test length(X[1]) == 2
        @test abs.((1/N) * sum(X)) ≤ [0.5, 0.5] # Check that the mean is close to zero  
    end

    @testset "Algebraic Reference" begin
        f(x) = exp(-sum(x.^2))
        model = PSDModel(Legendre(0.0..1.0)^2, 
                        :downward_closed, 3)
        sra = SelfReinforcedSampler(
            f,
            model,
            1, :Chi2U,
            PSDModels.AlgebraicReference{2, Float64}();
            N_sample=1000,
            maxit=6000,trace=true
        )

        N = 1000
        X = PSDModels.sample_reference(sra, N)
        @test length(X) == N
        @test length(X[1]) == 2
        @test abs.(sum(X))/N ≤ [0.5, 0.5] # Check that the mean is close to zero  
    end
end


@testset "self SelfReinforcedSampler" begin
    @testset "simple" begin
        f(x) = sum(x.^2 + x.^4)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2U,
            PSDModels.ScalingReference{2}(-ones(2), ones(2));
            N_sample=500,
        )
        for _=1:10
            x = PSDModels.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
    end

    @testset "indefinite domain" begin
        f(x) = exp(-0.1*sum(x.^2))
        model = PSDModel(Legendre(0.0..1.0)^2, 
                    :downward_closed, 3)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2U,
            PSDModels.GaussianReference{2, Float64}(2.0);
            relaxation_method=:algebraic,
            N_sample=1000
        )
        x = PSDModels.sample(sra)
    end

    @testset "broadcasted target" begin
        f(X) = map(x->exp(-sum(x.^2)), X)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 3)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2U,
            PSDModels.ScalingReference{2}(-ones(2), ones(2));
            N_sample=500,
            broadcasted_tar_pdf=true,
            trace=false,
        )
        for _=1:10
            x = PSDModels.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
    end

    @testset "broadcasted target indefinite domain" begin
        f(X) = map(x->exp(-sum(x.^2)), X)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 2)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2U,
            PSDModels.GaussianReference{2, Float64}(2.0);
            N_sample=500,
            broadcasted_tar_pdf=true,
            trace=false,
        )
        x = PSDModels.sample(sra)
    end

    @testset "Sampling irregular domain" begin
        f(x) = sin(sum(x))
        model = PSDModel(Legendre(0.0..1.0)^2, 
                    :downward_closed, 3)
        sra = SelfReinforcedSampler(
            f,
            model,
            1, :Chi2U,
            PSDModels.ScalingReference{2}([-1.0, 1.0], [0.0, 2.0]);
            relaxation_method=:algebraic,
            N_sample=1000,
            trace=false,
            λ_1=1e-3
        )
        x = PSDModels.sample(sra)
    end

    @testset "broadcasted irregular domain" begin
        f(X) = map(x->sin(sum(x)), X)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 2)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2U,
            PSDModels.ScalingReference{2}([-1.0, 1.0], [0.0, 2.0]);
            N_sample=500,
            broadcasted_tar_pdf=true,
            trace=false,
        )
        x = PSDModels.sample(sra)
    end
end

@testset "SelfReinforcedSampler different fit methods" begin
    @testset "Chi2" begin
        f(x) = sum(x.^2 + x.^4)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2,
            PSDModels.ScalingReference{2}(-ones(2), ones(2));
            N_sample=500,
        )
        for _=1:10
            x = PSDModels.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
    end

    @testset "KL" begin
        f(x) = sum(x.^2 + x.^4)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :KL,
            PSDModels.ScalingReference{2}(-ones(2), ones(2));
            N_sample=500,
        )
        for _=1:10
            x = PSDModels.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
    end

    @testset "Hellinger" begin
        f(x) = sum(x.^2 + x.^4)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Hellinger,
            PSDModels.ScalingReference{2}(-ones(2), ones(2));
            N_sample=500,
        )
        for _=1:10
            x = PSDModels.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
    end

    @testset "TV" begin
        f(x) = sum(x.^2 + x.^4)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :TV,
            PSDModels.ScalingReference{2}(-ones(2), ones(2));
            N_sample=500,
        )
        for _=1:10
            x = PSDModels.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
    end
end

# @testset "SubsetSampler" begin
#     @testset "simple training from samples"

#     end
# end