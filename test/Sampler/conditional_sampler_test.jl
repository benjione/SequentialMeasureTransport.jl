using Test
using SequentialMeasureTransport.BridgingDensities

@testset "simple PSDModel conditional sampler" begin
    f(x) = sum(x.^2)
    model = PSDModel(Legendre(-1.0..1.0)^2, :downward_closed, 2)
    X = [rand(2) * 2 .- 1 for _=1:500]
    Y = f.(X)
    fit!(model, X, Y, trace=false)
    SMT.normalize!(model)
    cond_sampler = ConditionalSampler(model, 1)
    Y_samp = SMT.cond_sample(cond_sampler, [[rand() * 2 - 1] for _=1:1000]; threading=false)
    @test all([-1≤xi[1]≤1 for xi in Y_samp])

    ## test pdf and marginal pdf
    Y_marg = [[rand() * 2 - 1] for _=1:100]
    Y = [rand(2) * 2 .- 1 for _=1:100]
    f_pdf(x) = 3/8 * f(x)
    f_marg_pdf(x) = 3/4 * x.^2 .+ 1/4
    @test norm(pdf.(Ref(cond_sampler), Y) - vcat(f_pdf.(Y)...), 2)/norm(vcat(f_pdf.(Y)...), 2) < 0.01
    @test norm(SMT.marg_pdf.(Ref(cond_sampler), Y_marg) - vcat(f_marg_pdf.(Y_marg)...), 2)/norm(vcat(f_pdf.(Y)...), 2) < 0.01

    ## test conditional pdf
    Y = [[rand() * 2 - 1] for _=1:100]
    f_cond_pdf(y, x) = f_pdf([x; y]) / f_marg_pdf(x)[1]
    for _=1:10
        x = [rand() * 2 - 1]
        vec1 = Y .|> y -> f_cond_pdf(y, x)
        vec2 = Y .|> y -> SMT.cond_pdf(cond_sampler, y, x)
        @test norm(vec1 - vec2, 2)/norm(vec1, 2) < 0.01
    end
end

@testset "2 layer conditional sampler" begin
    f(x) = sum(x.^2)
    model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 3)
    
    sra = SelfReinforcedSampler(f, model, 2, 
                :Chi2, SMT.ScalingReference{2, 1}(-ones(2), ones(2)); 
                N_sample=1000,
                dC=1)

    ## test pdf and marginal pdf
    Y_marg = [[rand() * 2 - 1] for _=1:100]
    Y = [rand(2) * 2 .- 1 for _=1:100]
    f_pdf(x) = 3/8 * f(x)
    f_marg_pdf(x) = 3/4 * x.^2 .+ 1/4
    @test norm(pdf.(Ref(sra), Y) - vcat(f_pdf.(Y)...), 2)/norm(vcat(f_pdf.(Y)...), 2) < 0.1
    @test norm(SMT.marg_pdf.(Ref(sra), Y_marg) - vcat(f_marg_pdf.(Y_marg)...), 2)/norm(vcat(f_marg_pdf.(Y)...), 2) < 0.1

    ## test conditional pdf
    Y = [[rand() * 2 - 1] for _=1:100]
    f_cond_pdf(y, x) = f_pdf([x; y]) / f_marg_pdf(x)[1]
    for _=1:10
        x = [rand() * 2 - 1]
        vec1 = Y .|> y -> f_cond_pdf(y, x)
        vec2 = Y .|> y -> SMT.cond_pdf(sra, y, x)
        @test norm(vec1 - vec2, 2)/norm(vec1, 2) < 0.1
    end
end


@testset "2 layer conditional sampler indefinite domain" begin
    @testset "Gaussian reference" begin
        f(x) = pdf(MvNormal(zeros(2), ones(2)), x)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 3)
        
        sra = SelfReinforcedSampler(f, model, 2, 
                    :Chi2, SMT.GaussianReference{2, 1, Float64}(2.0); 
                    N_sample=1000,
                    dC=1)

        ## test pdf and marginal pdf
        Y_marg = [[rand() * 2 - 1] for _=1:100]
        Y = [rand(2) * 2 .- 1 for _=1:100]
        f_pdf(x) = f(x)
        f_marg_pdf(x) = pdf(Normal(0, 1), x)
        @test norm(pdf.(Ref(sra), Y) - vcat(f_pdf.(Y)...), 2)/norm(vcat(f_pdf.(Y)...), 2) < 0.1
        @test norm(SMT.marg_pdf.(Ref(sra), Y_marg) - vcat(f_marg_pdf.(Y_marg)...), 2)/norm(vcat(f_marg_pdf.(Y)...), 2) < 0.1

        ## test conditional pdf
        Y = [[rand() * 2 - 1] for _=1:100]
        f_cond_pdf(y, x) = f_pdf([x; y]) / f_marg_pdf(x)[1]
        for _=1:10
            x = [rand() * 2 - 1]
            cond_pdf2 = SMT.cond_pushforward(sra, y->SMT.marg_Jacobian(SMT.GaussianReference{2, 1, Float64}(2.0), y), x)
            vec1 = Y .|> y -> f_cond_pdf(y, x)
            vec2 = Y .|> y -> SMT.cond_pdf(sra, y, x)
            vec3 = Y .|> y -> cond_pdf2(y)
            @test norm(vec1 - vec2, 2)/norm(vec1, 2) < 0.1
            @test norm(vec1 - vec3, 2)/norm(vec1, 2) < 0.1
        end
    end

    @testset "Algebraic reference" begin
        f(x) = pdf(MvNormal(zeros(2), 0.5*ones(2)), x)
        model = PSDModel(Legendre(0.0..1.0)^2, :downward_closed, 5)
        
        bridge = AlgebraicBridgingDensity{2}(f, [0.5, 1.0])
        sra = SelfReinforcedSampler(bridge, model, 2, 
                    :Chi2, SMT.AlgebraicReference{2, 1, Float64}(); 
                    N_sample=1000,
                    dC=1)

        ## test pdf and marginal pdf
        Y_marg = [[rand() * 2 - 1] for _=1:100]
        Y = [rand(2) * 2 .- 1 for _=1:100]
        f_pdf(x) = f(x)
        f_marg_pdf(x) = pdf(Normal(0, 0.5), x)
        @test norm(pdf.(Ref(sra), Y) - vcat(f_pdf.(Y)...), 2)/norm(vcat(f_pdf.(Y)...), 2) < 0.1
        @test norm(SMT.marg_pdf.(Ref(sra), Y_marg) - vcat(f_marg_pdf.(Y_marg)...), 2)/norm(vcat(f_marg_pdf.(Y)...), 2) < 0.1

        ## test conditional pdf
        Y = [[rand() * 2 - 1] for _=1:100]
        f_cond_pdf(y, x) = f_pdf([x; y]) / f_marg_pdf(x)[1]
        for _=1:10
            x = [rand() * 2 - 1]
            cond_pdf2 = SMT.cond_pushforward(sra, y->SMT.marg_Jacobian(SMT.AlgebraicReference{2, 1, Float64}(), y), x)
            vec1 = Y .|> y -> f_cond_pdf(y, x)
            vec2 = Y .|> y -> SMT.cond_pdf(sra, y, x)
            vec3 = Y .|> y -> cond_pdf2(y)
            @test norm(vec1 - vec2, 2)/norm(vec1, 2) < 0.1
            @test norm(vec1 - vec3, 2)/norm(vec1, 2) < 0.1
        end
    end
end

@testset "2D from data" begin
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
    ref_map = SequentialMeasureTransport.ReferenceMaps.GaussianReference{2, 1, Float64}(2.5)
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
