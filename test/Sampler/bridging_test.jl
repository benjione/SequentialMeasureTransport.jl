using PSDModels.ReferenceMaps
using PSDModels.BridgingDensities

@testset "Alpha-geodesic bridging test" begin
    @testset "KL alpha, α=1" begin
        f(x) = exp(-sum((x.-2.0).^2))
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 3)
        ref_map = GaussianReference{2, Float64}(2.0)
        bridging = AlphaGeodesicBridgingDensity{2}(
            1.0, x->pdf(ref_map, x), x->f(x), [0.05, 0.2, 0.5, 0.8, 1.0]
        )
        sra = SelfReinforcedSampler(
            bridging,
            model,
            5, :Chi2U,
            ref_map;
            N_sample=500,
            trace=false,
        )
        N = 1000
        X = PSDModels.sample(sra, N)
        @test length(X) == N
        @test length(X[1]) == 2
        # Check that the mean is close to 2
        @test [1.5, 1.5] ≤ sum(X)/N ≤ [2.5, 2.5]
    end

    @testset "KL dual alpha, α=0" begin
        f(x) = exp(-sum((x.-2.0).^2))
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 3)
        ref_map = GaussianReference{2, Float64}(2.0)
        bridging = AlphaGeodesicBridgingDensity{2}(
            0.0, x->pdf(ref_map, x), x->f(x), [0.05, 0.2, 0.5, 0.8, 1.0]
        )
        sra = SelfReinforcedSampler(
            bridging,
            model,
            5, :Chi2U,
            ref_map;
            N_sample=500,
            trace=false,
        )
        N = 1000
        X = PSDModels.sample(sra, N)
        @test length(X) == N
        @test length(X[1]) == 2
        # Check that the mean is close to 2
        @test [1.5, 1.5] ≤ sum(X)/N ≤ [2.5, 2.5]
    end
end