


@testset "Single Projection" begin
    f(x) = pdf(MvNormal(0.5.+zeros(2), 0.2*diagm(ones(2))), x)
    f_marg(x) = pdf(Normal(0.5, 0.2), x)[1]
    model = PSDModel(Legendre(0.0..1.0), :downward_closed, 3)
    X = rand(1, 1000)
    Y = map(x->f_marg(x)[1], eachcol(X))
    SMT.Chi2_fit!(model, eachcol(X), Y, trace=false)
    smp = Sampler(model)
    map1 = SMT.ReferenceMaps.GaussianReference{2, Float64}(2.0)
    map2 = SMT.ReferenceMaps.GaussianReference{1, Float64}(2.0)
    proj_map = SMT.ProjectionMapping{2, 0}(smp, [1], map1, map2)

    ## check that pullback is pushforward
    for i=1:100
        x = rand(2)
        @test isapprox(x, SMT.pullback(proj_map, SMT.pushforward(proj_map, x)), atol=1e-9)
    end

    ## check that the distribution pushforward is pullback
    pb_pf_func = SMT.pullback(proj_map, SMT.pushforward(proj_map, x->1.0))
    for i=1:100
        x = rand(2)
        @test isapprox(pb_pf_func(x), 1.0, atol=1e-9)
    end

    #compare pdf with marginal
    rng = [[x, y] for x in rand(20), y in rand(20)]
    rng = reshape(rng, length(rng))
    vec1 = pdf.(Ref(proj_map), rng)
    @test norm(vec1 - f_marg.(rng), 2)/norm(f_marg.(rng), 2) < 0.4
end

@testset "Single conditional projection" begin
    variance = 0.5
    scaling_1d = cdf(Normal(0.5, variance), 1.0) - cdf(Normal(0.5, variance), 0.0)
    rng = range(0, 1, length=200)
    scaling_2d = 1/length(rng)^2 * sum(pdf(MvNormal(0.5.+zeros(2), variance*diagm(ones(2))), [x, y]) for x in rng, y in rng)
    f(x) = pdf(MvNormal(0.5.+zeros(3), variance*diagm(ones(3))), x)
    f_marg(x) = (1/scaling_2d) * pdf(MvNormal(0.5.+zeros(2), variance*diagm(ones(2))), x)
    f_margin_single(x) = (1/scaling_1d) * pdf(Normal(0.5, variance), x[1])
    model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 5)
    X = rand(2, 1000)
    Y = map(x->f_marg(x), eachcol(X))
    SMT.Chi2_fit!(model, eachcol(X), Y, trace=false)
    smp = ConditionalSampler(model, 1)
    map1 = SMT.ReferenceMaps.GaussianReference{3, 1, Float64}(2.0)
    map2 = SMT.ReferenceMaps.GaussianReference{2, 1, Float64}(2.0)
    # map1 = SMT.ReferenceMaps.AlgebraicReference{3, Float64}()
    # map2 = SMT.ReferenceMaps.AlgebraicReference{2, Float64}()
    proj_map = SMT.ProjectionMapping{3, 1}(smp, [1, 3], map1, map2)

    ## check that pullback is pushforward
    for i=1:100
        x = rand(3)
        @test isapprox(x, SMT.pullback(proj_map, SMT.pushforward(proj_map, x)), atol=1e-9)
    end

    # ## check that the distribution pushforward is pullback
    pb_pf_func = SMT.pullback(proj_map, SMT.pushforward(proj_map, x->1.0))
    for i=1:100
        x = rand(3)
        @test isapprox(pb_pf_func(x), 1.0, atol=1e-9)
    end

    ## for the marginal
    for i=1:100
        x = rand(2)
        @test isapprox(x, SMT.marg_pullback(proj_map, SMT.marg_pushforward(proj_map, x)), atol=1e-9)
    end

    ## for the marginal
    pb_pf_func = SMT.marg_pullback(proj_map, SMT.marg_pushforward(proj_map, x->1.0))
    for i=1:100
        x = rand(2)
        @test isapprox(pb_pf_func(x), 1.0, atol=1e-9)
    end

    f_marg2(x) = f_marg([x[1], x[3]])

    #compare pdf with marginal
    rng = [[x, y, z] for x in rand(20), y in rand(20), z in rand(20)]
    rng = reshape(rng, length(rng))
    vec1 = pdf.(Ref(proj_map), rng)
    @test norm(vec1 - f_marg2.(rng), 2)/norm(f_marg2.(rng), 2) < 0.2

    # compare marginal with second marginal
    rng = [[x, y] for x in rand(20), y in rand(20)]
    rng = reshape(rng, length(rng))
    vec1 = SMT.marg_pdf.(Ref(proj_map), rng)
    @test norm(vec1 - f_margin_single.(rng), 2)/norm(f_margin_single.(rng), 2) < 0.2
end

@testset "Simple from data" begin
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
    ref_map = SMT.ReferenceMaps.GaussianReference{2, T}(T(2.0))
    to_subspace_ref_map = SMT.ReferenceMaps.GaussianReference{2, T}(T(2.0))
    subspace_ref_map = SMT.ReferenceMaps.GaussianReference{1, T}(T(2.0))

    # ref_map = SequentialMeasureTransport.ReferenceMaps.AlgebraicReference{2, T}()
    # to_subspace_ref_map = SequentialMeasureTransport.ReferenceMaps.AlgebraicReference{2, T}()
    # subspace_ref_map = SequentialMeasureTransport.ReferenceMaps.AlgebraicReference{1, T}()

    model = PSDModel{T}(Legendre(T(0)..T(1)), :downward_closed, 5)

    sra_sub = SMT.SelfReinforced_ML_estimation(eachcol(T.(X)), 
                    model, bridge, ref_map;
                    subspace_reference_map=subspace_ref_map,
                    to_subspace_reference_map=to_subspace_ref_map, 
                    trace=false)

    X_sample = SequentialMeasureTransport.sample(sra_sub, 100)
    @test all([length(x) == 2 for x in X_sample])
    # test pdf
    rng = [[x...] for x in Iterators.product(range(-5, 5, length=50), range(-5, 5, length=50))]
    rng = reshape(rng, length(rng))
    @test norm(pdf.(Ref(sra_sub), rng) - f.(rng), Inf) < 0.4

    ## test that pushforward is pullback
    for x in X_sample
        @test isapprox(x, SequentialMeasureTransport.pullback(sra_sub, SequentialMeasureTransport.pushforward(sra_sub, x)), atol=1e-9)
    end

    ## test that the distribution pushforward is pullback
    pb_pf_func = SMT.pullback(sra_sub, SMT.pushforward(sra_sub, x->1.0))
    for i=1:100
        x = rand(2)
        @test isapprox(pb_pf_func(x), 1.0, atol=1e-9)
    end
end

@testset "Conditional from data" begin
    distr1 = MvNormal([1.,1.,1.], diagm([0.2, 0.5, 0.5]))
    distr2 = MvNormal([-1.,-1.,-1.], diagm([0.5, 0.2, 0.5]))
    f1(x) = pdf(distr1, x)
    f2(x) = pdf(distr2, x)
    f(x) = (f1(x) + f2(x) )/ 2.0
    marg_distr1 = MvNormal([1.,1.], diagm([0.2, 0.5]))
    marg_distr2 = MvNormal([-1.,-1.], diagm([0.5, 0.2]))
    f_marg(x) = (pdf(marg_distr1, x) + pdf(marg_distr2, x) )/ 2.0
    f_cond(x, y) = f([x; y]) / f_marg(x)
    N = 2000
    N1 = rand(Binomial(N, 0.5))
    N2 = N - N1
    X1 = rand(distr1, N1)
    X2 = rand(distr2, N2)
    X = hcat(X1, X2)
    T = Float64
    bridge = DiffusionBrigdingDensity{2}(f, T[1.8, 1.5, 1.3, 1.2, 1.1, 1.0, 0.8, 0.75, 
                                            0.6, 0.5, 0.25, 0.18, 0.13,
                                            0.1, 0.07, 0.05, 0.02, 0.001, 0.007, 
                                            0.005, 0.003, 0.001, 0.0], T(1.0))
    # ref_map = SequentialMeasureTransport.ReferenceMaps.GaussianReference{3, T}(T(2.5))
    # to_subspace_ref_map = SequentialMeasureTransport.ReferenceMaps.GaussianReference{3, T}(T(3.0))
    # subspace_ref_map = SequentialMeasureTransport.ReferenceMaps.GaussianReference{2, T}(T(3.0))
    ref_map = SequentialMeasureTransport.ReferenceMaps.AlgebraicReference{3, 1, T}()
    to_subspace_ref_map = SequentialMeasureTransport.ReferenceMaps.AlgebraicReference{3, 1, T}()
    subspace_ref_map = SequentialMeasureTransport.ReferenceMaps.AlgebraicReference{2, 1, T}()

    model = PSDModel{T}(Legendre(T(0)..T(1))^2, :downward_closed, 3)

    sra_sub = SMT.SelfReinforced_ML_estimation(eachcol(T.(X)), 
                    model, bridge, ref_map;
                    subspace_reference_map=subspace_ref_map,
                    to_subspace_reference_map=to_subspace_ref_map,
                    dC=1,
                    dCsub=1,
                    trace=false)

    X_sample = SMT.sample(sra_sub, 100)
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
    @test norm(pdf.(Ref(sra_sub), rng) - f.(rng), 2)/norm(f.(rng), 2) < 0.4
    @test norm(SMT.marg_pdf.(Ref(sra_sub), rng_marg) - f_marg.(rng_marg), 2)/norm(f_marg.(rng_marg), 2) < 0.4
    

    ## check that conditional is normalized
    rng = range(-8, 8, 2000)
    Δrng = 16/2000
    for _=1:10
        x = if rand() < 0.5
            rand(marg_distr1)
        else
            rand(marg_distr2)
        end
        cond_pdf_func = SMT.cond_pushforward(sra_sub, y->SMT.Jacobian(SMT.AlgebraicReference{1, 0, Float64}(), y), x)
        int_cond = Δrng*sum(cond_pdf_func([y]) for y in rng)
        int_cond2 = Δrng*sum(SMT.cond_pdf(sra_sub, [y], x) for y in rng)
        @test isapprox(int_cond, 1.0, atol=0.05)
        @test isapprox(int_cond2, 1.0, atol=0.05)
    end

    # comparison of conditional difficult, use conditional negative log likelihood
    model_c_vec = rng .|> (x)->SMT.cond_pdf(sra_sub, x[3:3], x[1:2])
    c_vec = rng .|> (x)->f_cond(x[1:2], x[3:3])
    @test norm(model_c_vec - c_vec, 2)/norm(c_vec, 2) < 0.1
    X1 = rand(distr1, N1)
    X2 = rand(distr2, N2)
    X = hcat(X1, X2)
    cond_KL = (1/N) * sum(log.(f_cond(x[1:2], x[3:3])) for x in eachcol(X))

    cond_neg_log_likelihood = (1/N) * sum(log.(SMT.cond_pdf(sra_sub, x[3:3], x[1:2])) for x in eachcol(X))

    KL_cond = cond_KL - cond_neg_log_likelihood

    @assert KL_cond < 8.0


    ## test that pushforward is pullback
    for i=1:100
        x = rand(2)
        @test isapprox(x, SMT.marg_pullback(sra_sub, SMT.marg_pushforward(sra_sub, x)), atol=1e-9)
    end

    ## test that marginal pushforward is pullback
    pb_pf_func = SMT.marg_pullback(sra_sub, SMT.marg_pushforward(sra_sub, x->1.0))
    for i=1:100
        x = rand(2)
        @test isapprox(pb_pf_func(x), 1.0, atol=1e-9)
    end

    ## test that conditional pushforward is pullback
    for k=1:10
        x = rand(2)
        for i=1:100
            y = rand(1)
            y_pf = SMT.cond_pushforward(sra_sub, y, x)
            @test isapprox(y, SMT.cond_pullback(sra_sub, y_pf, x), atol=1e-9)
        end

        pb_pf_func = SMT.cond_pullback(sra_sub, SMT.cond_pushforward(sra_sub, x->1.0, x), x)
        for i=1:100
            y = rand()
            @test isapprox(pb_pf_func([y]), 1.0, atol=1e-9)
        end
    end
end


@testset "Marginal mapping with unused dimension" begin
    @testset "Simple" begin
        f(x) = pdf(Normal(0.0, 1.0), x)
        model = PSDModel(Legendre(-5.0..5.0), :downward_closed, 4)
        X = rand(1, 1000)
        Y = map(x->f(x)[1], eachcol(X))
        SequentialMeasureTransport.Chi2_fit!(model, eachcol(X), Y, trace=false)
        normalize!(model)
        smp = Sampler(model)

        smp_marg = SequentialMeasureTransport.MarginalMapping{2, 0}(smp, [1])

        # compare pdfs
        f_app = SequentialMeasureTransport.pushforward(smp, x->1.0)
        f_app_marg = SequentialMeasureTransport.pushforward(smp_marg, x->1.0)

        rng = range(-5, 5, length=100)
        for x in rng
            @test isapprox(f_app([x]), f_app_marg([x, 0.0]), atol=1e-9)
        end
    end
    @testset "self reinforced" begin
        f1(x, y) = pdf(MvNormal([0.0, 0.0], [1.0 0.8; 0.8 1.0]), [x, y])
        f(x, y, z) = f1(x, y) * f2(y, z)
    
        f1(x) = f1(x[1], x[2])
        f2(x) = f2(x[1], x[2])
        f(x) = f(x[1], x[2], x[3])
    
        ref_map = SequentialMeasureTransport.ReferenceMaps.ScalingReference(-10.0*ones(2), 10.0*ones(2))
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 3)
    
        fit_method(m, x, y; kwargs...) = SequentialMeasureTransport.α_divergence_fit!(m, 2.0, x, y; kwargs...)
    
        sra = SelfReinforcedSampler(f1, model, 3, :Chi2,
                            ref_map; trace=false,
                            ϵ=1e-6, λ_2=0.0, λ_1=0.0,
                            algebraic_base=2.0,
                            N_sample=1000,
                            # optimizer=Hypatia.Optimizer
                )
        sra_marg = SequentialMeasureTransport.MarginalMapping{3, 0}(sra, [1,2])
        pdf_func = SequentialMeasureTransport.pushforward(sra_marg, x->1.0)
        pdf_func_orig = SequentialMeasureTransport.pushforward(sra, x->1.0)
        rng = [[x...] for x in Iterators.product(range(-5, 5, length=25), range(-5, 5, length=25))]
        rng = reshape(rng, length(rng))
        for x in rng
            @test isapprox(pdf_func([x; [0.0]]), pdf_func_orig(x), atol=1e-9)
        end 
    end
end