using PSDModels.GraphicalModels

@testset "Graphical model from couplings" begin
    Σ = [1.0 0.5; 0.5 1.0]
    Λ_1 = [4/3 0.0 -2/3; 0.0 0.0 0.0; -2/3 0.0 4/3]
    Λ_2 = [4/3 -2/3 0.0; -2/3 4/3 0.0; 0.0 0.0 0.0]
    Λ_12 = Λ_1 + Λ_2
    f(x) = pdf(MvNormal(zeros(3), inv(Λ_12)), x)
    f_1(x) = pdf(MvNormal(zeros(2), Σ), x)
    f_2(x) = pdf(MvNormal(zeros(2), Σ), x)

    T = Float64
    Fac1 = PSDModels.GraphicalModels.Factor{T}([1, 3], f_1)
    Fac2 = PSDModels.GraphicalModels.Factor{T}([1, 2], f_2)
    g_model = PSDModels.GraphicalModels.GraphicalModel([Fac1, Fac2])

    function model_factory(d::Int)
        return PSDModels.PSDModel(Legendre(zero(T)..one(T))^d, :downward_closed, 5)
    end

    ref_map = PSDModels.ReferenceMaps.AlgebraicReference{3, T}()
    fit_method! = PSDModels.Statistics.Chi2_fit!
    graph_model = PSDModels.GraphicalModels.GraphSampler(g_model, model_factory, 
                        fit_method!, ref_map, ref_map, trace=false)

    rng1 = range(-5, 5, length=50)
    rng2 = range(-5, 5, length=50)
    rng3 = range(-5, 5, length=50)
    app_vec = Iterators.product(rng1, rng2, rng3) .|> collect .|> (x) -> pdf(graph_model, x)
    real_vec = Iterators.product(rng1, rng2, rng3) .|> collect .|> (x) -> f(x)

    @test mean((app_vec .- real_vec).^2) < 1e-3
end


@testset "Time series model" begin
    T = Float64
    Fmap(x, t, Θ) = x*exp(-Θ*t)
    Δt = 0.2
    timesteps = 0.0:Δt:1.0
    N = length(timesteps)
    x0 = 0.0
    Θ_true = 3.0
    Y = Fmap.(Ref(x0), timesteps[2:end], Ref(Θ_true)) .+ rand(Normal(0.0, 0.1), length(timesteps)-1)
    f(x_state, x_state_prev, param) = pdf(Normal(x_state_prev[1], 0.1), x_state[1])
    L(y, x_state, param) = pdf(Normal(y[1], 0.1), x_state[1])
    # L(y, x_state, param) = 1.0
    π_X(x_state_prev) = pdf(Normal(x0, 0.1), x_state_prev[1])
    # π_Θ(x) = pdf(Normal(Θ_true, 1.0), x[1])
    π_Θ(param) = 1.0
    model = PSDModels.GraphicalModels.TimeSeriesModel{T}(π_X,π_Θ, f, L, 1, 1, 0, N)
    function model_factory(d::Int)
        return PSDModels.PSDModel(Legendre(zero(T)..one(T))^d, :downward_closed, 4)
    end
    d = N
    ref_map = PSDModels.ReferenceMaps.AlgebraicReference{d, T}()
    # ref_map = PSDModels.ReferenceMaps.ScalingReference{d}(-ones(d), ones(d))
    fit_method! = (m, x, y; kwargs...)->PSDModels.Statistics.α_divergence_fit!(m, 2.0, x, y; kwargs...)
    # fit_method! = PSDModels.Statistics.Chi2_fit!
    Y = sin.(timesteps[2:end])
    smp = PSDModels.GraphicalModels.TimeSeriesSampler(model, eachcol(Y), model_factory, 
                    fit_method!, ref_map, ref_map, trace=true, N_sample=4000)
    # test if parameter estimation close to true value
    PSDModels.marg_pushforward(smp, PSDModels.sample_reference(smp)[1:1]; layers=1:10)
    PSDModels.marg_pdf(smp, [10.0])
    X = PSDModels.pushforward(smp, PSDModels.sample_reference(smp), layers=1:4)
    X = rand(smp, 100)
    X = reduce(hcat, X)[1,:]
    mean(X)
    X2 = PSDModels.marg_sample(smp, 100, threading=true)
    X2 = reduce(hcat, X2)[1,:]
    mean(X2)
    for i=1:N
        print(mean(X[i,:]), ", ")
    end
    using Plots
    plt = plot(rand(smp))
    for i=1:5
        plt = plot!(rand(smp), label=nothing)
    end
    plt
end