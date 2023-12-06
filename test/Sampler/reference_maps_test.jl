@testset "Scaling Reference" begin
    f(x) = sum(x.^2)
    model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 1)
    sra = SelfReinforcedSampler(
        f,
        model,
        1, :Chi2,
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
        1, :Chi2,
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
        1, :Chi2,
        PSDModels.AlgebraicReference{2, Float64}();
        N_sample=1000,
        maxit=6000,trace=false
    )

    N = 1000
    X = PSDModels.sample_reference(sra, N)
    @test length(X) == N
    @test length(X[1]) == 2
    @test abs.(sum(X))/N ≤ [0.5, 0.5] # Check that the mean is close to zero  
end