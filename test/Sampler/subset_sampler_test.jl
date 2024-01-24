

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
                    dC=1,
                    dCsub=1,
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

@testset "add layer marginal mapping" begin
    
    f1(x, y) = pdf(MvNormal([0.0, 0.0], [1.0 0.8; 0.8 1.0]), [x, y])
    f2(y, z) = pdf(MvNormal([0.0, 0.0], [1.0 -0.8; -0.8 1.0]), [y, z])
    f(x, y, z) = f1(x, y) * f2(y, z)

    f1(x) = f1(x[1], x[2])
    f2(x) = f2(x[1], x[2])
    f(x) = f(x[1], x[2], x[3])

    ref_map = PSDModels.ReferenceMaps.AlgebraicReference{3, Float64}()
    model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 3)

    fit_method(m, x, y; kwargs...) = PSDModels.α_divergence_fit!(m, 2.0, x, y; kwargs...)

    # create empty SelfReinforcedSampler
    sample = PSDModels.CondSampler{3, 0, Float64}(ref_map, ref_map)

    ### first f1 than f2
    # initialize β
    β = (1/16, 1/16)

    PSDModels.add_layer!(sample, (x)->f1(x)^β[1], model, 
                    fit_method, [1, 2]; variable_ordering=[2, 1])

    β = (1/8, 1/16)
    PSDModels.add_layer!(sample, (x)->f1(x)^β[1], model, 
                    fit_method, [1, 2]; variable_ordering=[2, 1])

    β = (1/4, 1/16)
    PSDModels.add_layer!(sample, (x)->f1(x)^β[1], model, 
                    fit_method, [1, 2]; variable_ordering=[2, 1])

    β = (1/2, 1/16)
    PSDModels.add_layer!(sample, (x)->f1(x)^β[1], model, 
                    fit_method, [1, 2]; variable_ordering=[2, 1])

    β = (1, 1/16)
    PSDModels.add_layer!(sample, (x)->f1(x)^β[1], model, 
                    fit_method, [1, 2]; variable_ordering=[2, 1])

    β = (1, 1/16)
    PSDModels.add_layer!(sample, (x)->f2(x)^β[2], model, 
                    fit_method, [2, 3]; variable_ordering=[1, 2])

    β = (1, 1/8)
    PSDModels.add_layer!(sample, (x)->f2(x)^β[2], model, 
                    fit_method, [2, 3]; variable_ordering=[1, 2])

    β = (1, 1/4)
    PSDModels.add_layer!(sample, (x)->f2(x)^β[2], model, 
                    fit_method, [2, 3]; variable_ordering=[1, 2])

    β = (1, 1/2)
    PSDModels.add_layer!(sample, (x)->f2(x)^β[2], model, 
                    fit_method, [2, 3]; variable_ordering=[1, 2])

    β = (1, 1)
    PSDModels.add_layer!(sample, (x)->f2(x)^β[2], model, 
                    fit_method, [2, 3]; variable_ordering=[1, 2])
end