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
        model = PSDModel(Legendre()^2, :downward_closed, 2)
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
end