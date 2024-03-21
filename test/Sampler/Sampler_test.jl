using SequentialMeasureTransport.ReferenceMaps
using SequentialMeasureTransport.BridgingDensities
using ApproxFun
using Distributions

@testset "Building large sampler" begin
    for d=2:10
        @testset "$(d) dimension" begin
            f(x) = sum(x.^2)
            model = PSDModel(Legendre(-1.0..1.0)^d, :downward_closed, 1)
            X = [rand(d) * 2 .- 1 for _=1:500]
            Y = f.(X)
            fit!(model, X, Y, trace=false)
            smp = Sampler(model)
            x = SMT.sample(smp)
            @test all([-1≤xi≤1 for xi in x])
        end
    end
end

@testset "Reference map and reference sampling" begin
    include("reference_maps_test.jl")
end

@testset "Bridging methods test" begin
    include("bridging_test.jl")
end


@testset "self SelfReinforcedSampler" begin
    @testset "simple" begin
        f(x) = 1/(4*(16/15)) * sum(x.^2 + x.^4)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 2)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2,
            SMT.ScalingReference{2}(-ones(2), ones(2));
            N_sample=500,
        )
        for _=1:10
            x = SMT.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
        N = 20
        # test close to each other
        rng = [[x...] for x in Iterators.product(
                range(-1, 1, length=N),
                range(-1, 1, length=N)
            )
        ]
        rng = reshape(rng, length(rng))
        @test norm(pdf.(Ref(sra), rng) - f.(rng), 2)/norm(f.(rng), 2) < 0.3
    end

    @testset "indefinite domain" begin
        f(x) = pdf(MvNormal([0.0, 0.0], 4.0*diagm(ones(2))), x)
        model = PSDModel(Legendre(0.0..1.0)^2, 
                    :downward_closed, 3)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2,
            SMT.GaussianReference{2, Float64}(2.0);
            relaxation_method=:algebraic,
            N_sample=1000
        )
        x = SMT.sample(sra)
        N = 20
        rng = [[x...] for x in Iterators.product(
                range(-1, 1, length=N),
                range(-1, 1, length=N)
            )
        ]
        rng = reshape(rng, length(rng))
        @test norm(pdf.(Ref(sra), rng) - f.(rng), 2)/norm(f.(rng), 2) < 0.4
    end

    @testset "broadcasted target" begin
        f(X) = map(x->exp(-sum(x.^2)), X)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 3)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2,
            SMT.ScalingReference{2}(-ones(2), ones(2));
            N_sample=500,
            broadcasted_tar_pdf=true,
            trace=false,
        )
        for _=1:10
            x = SMT.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
    end

    @testset "broadcasted target indefinite domain" begin
        f(X) = map(x->exp(-sum(x.^2)), X)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 2)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2,
            SMT.GaussianReference{2, Float64}(2.0);
            N_sample=500,
            broadcasted_tar_pdf=true,
            trace=false,
        )
        x = SMT.sample(sra)
    end

    @testset "Sampling irregular domain" begin
        f(x) = sin(sum(x))
        model = PSDModel(Legendre(0.0..1.0)^2, 
                    :downward_closed, 3)
        sra = SelfReinforcedSampler(
            f,
            model,
            1, :Chi2,
            SMT.ScalingReference{2}([-1.0, 1.0], [0.0, 2.0]);
            relaxation_method=:algebraic,
            N_sample=1000,
            trace=false,
            λ_1=1e-3
        )
        x = SMT.sample(sra)
    end

    @testset "broadcasted irregular domain" begin
        f(X) = map(x->sin(sum(x)), X)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 2)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2,
            SMT.ScalingReference{2}([-1.0, 1.0], [0.0, 2.0]);
            N_sample=500,
            broadcasted_tar_pdf=true,
            trace=false,
        )
        x = SMT.sample(sra)
    end
end

@testset "SelfReinforcedSampler different fit methods" begin
    @testset "Chi2" begin
        f(x) = 1/(4*(16/15)) * sum(x.^2 + x.^4)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2,
            SMT.ScalingReference{2}(-ones(2), ones(2));
            N_sample=500,
        )
        for _=1:10
            x = SMT.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
        N = 20
        # test close to each other
        rng = [[x...] for x in Iterators.product(
                range(-1, 1, length=N),
                range(-1, 1, length=N)
            )
        ]
        rng = reshape(rng, length(rng))
        @test norm(pdf.(Ref(sra), rng) - f.(rng), 2)/norm(f.(rng), 2) < 0.3
    end

    @testset "KL" begin
        f(x) = 1/(4*(16/15)) * sum(x.^2 + x.^4)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :KL,
            SMT.ScalingReference{2}(-ones(2), ones(2));
            N_sample=500,
            trace=false,
            λ_2 = 0.0
        )
        for _=1:10
            x = SMT.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
        # do not test with test solver, since inaccurate for KL
        # N = 20
        # # test close to each other
        # rng = [[x...] for x in Iterators.product(
        #         range(-1, 1, length=N),
        #         range(-1, 1, length=N)
        #     )
        # ]
        # rng = reshape(rng, length(rng))
        # @test norm(pdf.(Ref(sra), rng) - f.(rng), 2)/norm(f.(rng), 2) < 0.3
    end

    @testset "Hellinger" begin
        f(x) = 1/(4*(16/15)) * sum(x.^2 + x.^4)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Hellinger,
            SMT.ScalingReference{2}(-ones(2), ones(2));
            N_sample=1000,
            trace=false,
            data_normalization=false,
        )
        for _=1:10
            x = SMT.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
        N = 20
        # test close to each other
        rng = [[x...] for x in Iterators.product(
                range(-1, 1, length=N),
                range(-1, 1, length=N)
            )
        ]
        rng = reshape(rng, length(rng))
        @test norm(pdf.(Ref(sra), rng) - f.(rng), 2)/norm(f.(rng), 2) < 0.3
    end

    @testset "TV" begin
        f(x) = 1/(4*(16/15)) * sum(x.^2 + x.^4)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 1)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :TV,
            SMT.ScalingReference{2}(-ones(2), ones(2));
            N_sample=500,
        )
        for _=1:10
            x = SMT.sample(sra)
            @test all([-1≤xi≤1 for xi in x])
        end 
        N = 20
        # test close to each other
        rng = [[x...] for x in Iterators.product(
                range(-1, 1, length=N),
                range(-1, 1, length=N)
            )
        ]
        rng = reshape(rng, length(rng))
        @test norm(pdf.(Ref(sra), rng) - f.(rng), 2)/norm(f.(rng), 2) < 0.3
    end
end


@testset "SRA ML estimation" begin
    include("SR_ML_test.jl")
end

@testset "Conditional Sampler" begin
    include("conditional_sampler_test.jl")
end

@testset "SubsetSampler" begin
   include("subset_sampler_test.jl") 
end
