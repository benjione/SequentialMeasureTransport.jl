

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
        ref_map = SequentialMeasureTransport.ReferenceMaps.GaussianReference{1, Float64}(2.0)
        sra = SequentialMeasureTransport.SelfReinforced_ML_estimation(
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

@testset "adaptive ML test" begin
    @testset "1D double gaussian" begin
        distr1 = Normal(-1.0, 0.5)
        distr2 = Normal(1.0, 0.5)
        distr = MixtureModel([distr1, distr2], [0.5, 0.5])
        f(x) = pdf(distr, x)
        X_train = rand(distr, 1000) #+ 0.05 * randn(500)
        X_val = rand(distr, 1000) #+ 0.05 * randn(200)
        X_train = [[x] for x in X_train]
        X_val = [[x] for x in X_val]
        model = PSDModel(Legendre(0.0..1.0), :downward_closed, 5)
        ref_map = SequentialMeasureTransport.ReferenceMaps.AlgebraicReference{1, Float64}()
        sra = SequentialMeasureTransport.Adaptive_Self_reinforced_ML_estimation(
            X_train,
            X_val,
            model,
            2.0,
            ref_map;
            # optimizer=Hypatia.Optimizer,
            trace=false,
            ϵ=1e-5,
        )

        # check neg log likelihood of test data
        neg_log_likelihood = (1/length(X_val)) * mapreduce(x->-log(pdf(sra, x)), +, X_val)

        @test neg_log_likelihood < 2.0
    end
end