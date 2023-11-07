

@testset "Simple ML" begin
    distr1 = MvNormal([2.,2.], diagm([0.1, 0.5]))
    distr2 = MvNormal([-2.,-2.], diagm([0.5, 0.1]))
    f1(x) = pdf(distr1, x)
    f2(x) = pdf(distr2, x)
    f(x) = (f1(x) + f2(x) )/ 2.0
    N = 1000
    N1 = rand(Binomial(N, 0.5))
    N2 = N - N1
    X1 = rand(distr1, N1)
    X2 = rand(distr2, N2)
    X = hcat(X1, X2)
    T = Float64
    bridge = DiffusionBrigdingDensity{1}(f, T[1.5, 1.3, 1.0, 0.8, 0.75, 0.5, 0.3, 0.25, 0.18, 0.13, 0.1, 0.07, 0.02, 0.01, 0.005, 0.0], T(2.0))
    ref_map = PSDModels.ReferenceMaps.GaussianReference{2, T}(T(2.0))
    to_subspace_ref_map = PSDModels.ReferenceMaps.GaussianReference{2, T}(T(2.0))
    subspace_ref_map = PSDModels.ReferenceMaps.GaussianReference{1, T}(T(2.0))

    model = PSDModel{T}(Legendre(T(0)..T(1)), :downward_closed, 5)

    sra_sub = PSDModels.SelfReinforced_ML_estimation(eachcol(T.(X)), 
                    model, bridge, ref_map;
                    subspace_reference_map=subspace_ref_map,
                    to_subspace_reference_map=to_subspace_ref_map, 
                    trace=false)

    X_sample = PSDModels.sample(sra_sub, 100)
    @test all([length(x) == 2 for x in X_sample])
    # test pdf
    rng = [[x...] for x in Iterators.product(range(-5, 5, length=50), range(-5, 5, length=50))]
    rng = reshape(rng, length(rng))
    @test norm(pdf.(Ref(sra_sub), rng) - f.(rng), Inf) < 0.4
end

@testset "Conditional" begin
    distr1 = MvNormal([2.,2.,2.], diagm([0.1, 0.5, 0.5]))
    distr2 = MvNormal([-2.,-2.,-2.], diagm([0.5, 0.1, 0.5]))
    f1(x) = pdf(distr1, x)
    f2(x) = pdf(distr2, x)
    f(x) = (f1(x) + f2(x) )/ 2.0
    marg_distr1 = MvNormal([2.,2.], diagm([0.1, 0.5]))
    marg_distr2 = MvNormal([-2.,-2.], diagm([0.5, 0.1]))
    f_marg(x) = (pdf(marg_distr1, x) + pdf(marg_distr2, x) )/ 2.0
    f_cond(x, y) = f([x; y]) / f_marg(x)
    N = 1000
    N1 = rand(Binomial(N, 0.5))
    N2 = N - N1
    X1 = rand(distr1, N1)
    X2 = rand(distr2, N2)
    X = hcat(X1, X2)
    T = Float64
    bridge = DiffusionBrigdingDensity{2}(f, T[1.5, 1.3, 1.0, 0.8, 0.75, 
                                            0.5, 0.3, 0.25, 0.18, 0.13, 
                                            0.1, 0.07, 0.02, 0.01, 0.005, 0.0], T(2.0))
    ref_map = PSDModels.ReferenceMaps.GaussianReference{3, T}(T(2.5))
    to_subspace_ref_map = PSDModels.ReferenceMaps.GaussianReference{3, T}(T(3.0))
    subspace_ref_map = PSDModels.ReferenceMaps.GaussianReference{2, T}(T(3.0))
    # to_subspace_ref_map = PSDModels.ReferenceMaps.AlgebraicReference{3, T}()
    # subspace_ref_map = PSDModels.ReferenceMaps.AlgebraicReference{2, T}()

    model = PSDModel{T}(Legendre(T(0)..T(1))^2, :downward_closed, 3)

    sra_sub = PSDModels.SelfReinforced_ML_estimation(eachcol(T.(X)), 
                    model, bridge, ref_map;
                    subspace_reference_map=subspace_ref_map,
                    to_subspace_reference_map=to_subspace_ref_map,
                    amount_cond_variable=1,
                    amount_reduced_cond_variables=1,
                    trace=false)

    X_sample = PSDModels.sample(sra_sub, 100)
    @test all([length(x) == 3 for x in X_sample])
    # test pdf
    N = 20
    rng = [[x...] for x in Iterators.product(
                range(-5, 5, length=N),
                range(-5, 5, length=N),
                range(-5, 5, length=N)
            )
        ]
    rng_marg = [[x...] for x in Iterators.product(
                range(-5, 5, length=N),
                range(-5, 5, length=N)
            )
        ]
    rng = reshape(rng, length(rng))
    rng_marg = reshape(rng_marg, length(rng_marg))
    @test norm(pdf.(Ref(sra_sub), rng) - f.(rng), Inf) < 0.4
    @test norm(PSDModels.marg_pdf.(Ref(sra_sub), rng_marg) - f_marg.(rng_marg), Inf) < 0.4
    model_c_vec = rng .|> (x)->PSDModels.cond_pdf(sra_sub, x[3:3], x[1:2])
    c_vec = rng .|> (x)->f_cond(x[1:2], x[3:3])
    @test (1/N^3)*norm(model_c_vec - c_vec, 2) < 0.1
end