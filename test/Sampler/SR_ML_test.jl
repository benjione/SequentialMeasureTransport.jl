

@testset "SRA ML" begin
    @testset "1D double gaussian" begin
        distr1 = Normal(0.0, 1.0)
        distr2 = Normal(2.0, 1.0)
        f(x) = 0.5 * pdf(distr1, x) + 0.5 * pdf(distr2, x)
        sample1 = rand(distr1, 500)
        sample2 = rand(distr2, 500)
        samples = vcat(sample1, sample2)
        samples = [[x] for x in samples]
        model = PSDModel(Legendre(0.0..1.0), :downward_closed, 3)
        bridge = DiffusionBrigdingDensity{1}(x->1.0, [1.0, 0.5, 0.25, 0.1, 0.05, 0.0], 2.0)
        ref_map = PSDModels.ReferenceMaps.GaussianReference{1, Float64}(2.0)
        sra = PSDModels.SelfReinforced_ML_estimation(
            samples,
            model,
            bridge,
            ref_map;
            # optimizer=Hypatia.Optimizer,
            trace=false,
        )
        # test densities are close
        for x in range(-5, 5, length=100)
            @test abs(pdf(sra, [x]) - f(x)) < 0.1
        end
    end
end