using PSDModels.ReferenceMaps
using PSDModels.BridgingDensities
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
            sampler = Sampler(model)
            x = PSDModels.sample(sampler)
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

    @testset "indefinite domain" begin
        f(x) = exp(-0.1*sum(x.^2))
        model = PSDModel(Legendre(0.0..1.0)^2, 
                    :downward_closed, 3)
        sra = SelfReinforcedSampler(
            f,
            model,
            2, :Chi2,
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
            2, :Chi2,
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
            2, :Chi2,
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
            1, :Chi2,
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
            2, :Chi2,
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


@testset "SRA ML estimation" begin
    include("SR_ML_test.jl")
end


@testset "SubsetSampler" begin
   include("subset_sampler_test.jl") 
end

@testset "Conditional Sampler" begin
    include("conditional_sampler_test.jl")
end

@testset "Sampling graphical model" begin
    include("graphical_model_test.jl")
end