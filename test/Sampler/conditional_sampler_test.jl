using Test
using SequentialMeasureTransport.BridgingDensities

@testset "PSDModel conditional sampler" begin
    f(x) = sum(x.^2)
    model = PSDModel(Legendre(-1.0..1.0)^2, :downward_closed, 2)
    X = [rand(2) * 2 .- 1 for _=1:500]
    Y = f.(X)
    fit!(model, X, Y, trace=false)
    SequentialMeasureTransport.normalize!(model)
    cond_sampler = ConditionalSampler(model, 1)
    Y = SequentialMeasureTransport.cond_sample(cond_sampler, [[rand() * 2 - 1] for _=1:1000]; threading=false)
    @test all([-1≤xi[1]≤1 for xi in Y])
end

@testset "Conditional SelfReinforcedSampler" begin
    @testset "ML 2D" begin
        distr1 = MvNormal([-1.0, 1.0], diagm([1.0, 1.0]))
        distr2 = MvNormal([1.0, 0.5], diagm([1.0, 1.0]))
        f(x) = 0.5 * pdf(distr1, x) + 0.5 * pdf(distr2, x)
        marg_distr1 = Normal(-1.0, 1.0)
        marg_distr2 = Normal(1.0, 1.0)
        f_marg(x) = 0.5 * pdf(marg_distr1, x) + 0.5 * pdf(marg_distr2, x)
        sample1 = rand(distr1, 600)
        sample2 = rand(distr2, 600)
        _samples = hcat(sample1, sample2)
        samples = [x for x in eachcol(_samples)]
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 3)
        bridge = DiffusionBrigdingDensity{2}(x->1.0, [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.0], 2.5)
        ref_map = SequentialMeasureTransport.ReferenceMaps.GaussianReference{2, Float64}(2.5)
        sra = SequentialMeasureTransport.SelfReinforced_ML_estimation(
            samples,
            model,
            bridge,
            ref_map;
            # optimizer=Hypatia.Optimizer,
            trace=false,
            dC=1,
        )
        # test densities are close
        for x in range(-5.0, 5.0, length=20)
            for y in range(-5.0, 5.0, length=20)
                @test abs(pdf(sra, [x, y]) - f([x, y])) < 0.1
            end
        end
        for x in range(-4.0, 4.0, length=100)
            @test abs(SequentialMeasureTransport.marg_pdf(sra, [x]) - f_marg(x)) < 0.2
        end
    end
end