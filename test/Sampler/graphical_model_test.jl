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
        return PSDModels.PSDModel(Legendre(zero(T)..one(T))^d, :downward_closed, 4)
    end

    ref_map = PSDModels.ReferenceMaps.AlgebraicReference{3, T}()
    fit_method! = PSDModels.Statistics.Chi2_fit!
    graph_model = PSDModels.GraphicalModels.GraphSampler(g_model, model_factory, fit_method!, ref_map, ref_map)

    rng1 = range(-5, 5, length=50)
    rng2 = range(-5, 5, length=50)
    rng3 = range(-5, 5, length=50)
    app_vec = Iterators.product(rng1, rng2, rng3) .|> collect .|> (x) -> pdf(graph_model, x)
    real_vec = Iterators.product(rng1, rng2, rng3) .|> collect .|> (x) -> f(x)
    @test mean((app_vec .- real_vec).^2) < 1e-3
end