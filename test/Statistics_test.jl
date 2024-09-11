using SequentialMeasureTransport.Statistics
using Distributions


@testset "Minimizer test" begin
    @testset "Chi2" begin
        f(x) = pdf(MvNormal([0.0, 0.0], 4.0*diagm(ones(2))), x)
        model = PSDModel(Legendre(-10.0..10.0)^2, :downward_closed, 5)
        # generate some data
        X = [(rand(2) * 20.0 .- 10.0) for i in 1:500]
        Y = f.(X)
        Chi2_fit!(model, X, Y, trace=false)
        normalize!(model)

        @test mean((model.(X) .- Y).^2) < 1e-3
    end

    @testset "KL" begin
        f(x) = pdf(MvNormal([0.0, 0.0], 4.0*diagm(ones(2))), x)
        model = PSDModel(Legendre(-10.0..10.0)^2, :downward_closed, 5)
        # generate some data
        X = [(rand(2) * 20.0 .- 10.0) for i in 1:500]
        Y = f.(X)
        Statistics.KL_fit!(model, X, Y; trace=false, use_putinar=false)
        normalize!(model)

        @test mean((model.(X) .- Y).^2) < 1e-3
    end

    @testset "KL Manopt" begin
        f(x) = pdf(MvNormal([0.0, 0.0], 4.0*diagm(ones(2))), x)
        model = PSDModel(Legendre(-10.0..10.0)^2, :downward_closed, 5)
        # generate some data
        X = [(rand(2) * 20.0 .- 10.0) for i in 1:500]
        Y = f.(X)
        Statistics.KL_fit!(model, X, Y, trace=false, SDP_library=:Manopt)
        normalize!(model)

        @test mean((model.(X) .- Y).^2) < 1e-3
    end

    @testset "reversed KL" begin
        f(x) = pdf(MvNormal([0.0, 0.0], 4.0*diagm(ones(2))), x)
        model = PSDModel(Legendre(-10.0..10.0)^2, :downward_closed, 5)
        # generate some data
        X = [(rand(2) * 20.0 .- 10.0) for i in 1:500]
        Y = f.(X)
        reversed_KL_fit!(model, X, Y, trace=false, use_putinar=false)
        normalize!(model)

        @test mean((model.(X) .- Y).^2) < 1e-3
    end

    @testset "TV" begin
        f(x) = pdf(MvNormal([0.0, 0.0], 4.0*diagm(ones(2))), x)
        model = PSDModel(Legendre(-10.0..10.0)^2, :downward_closed, 5)
        # generate some data
        X = [(rand(2) * 20.0 .- 10.0) for i in 1:500]
        Y = f.(X)
        TV_fit!(model, X, Y, trace=false)
        normalize!(model)

        @test mean((model.(X) .- Y).^2) < 1e-3
    end

    @testset "Hellinger" begin
        f(x) = pdf(MvNormal([0.0, 0.0], 4.0*diagm(ones(2))), x)
        model = PSDModel(Legendre(-10.0..10.0)^2, :downward_closed, 5)
        # generate some data
        X = [(rand(2) * 20.0 .- 10.0) for i in 1:500]
        Y = f.(X)
        Hellinger_fit!(model, X, Y,
                    trace=false, use_putinar=false)
        normalize!(model)

        @test mean((model.(X) .- Y).^2) < 1e-3
    end

    @testset "α-divergences" begin
        @testset "α=0.5" begin
            f(x) = pdf(MvNormal([0.0, 0.0], 4.0*diagm(ones(2))), x)
            model = PSDModel(Legendre(-10.0..10.0)^2, :downward_closed, 5)
            # generate some data
            X = [(rand(2) * 20.0 .- 10.0) for i in 1:500]
            Y = f.(X)
            α_divergence_fit!(model, 0.5, X, Y, trace=false, 
                        use_putinar=false)
            normalize!(model)

            @test mean((model.(X) .- Y).^2) < 1e-3
        end
        @testset "α=2" begin
            f(x) = pdf(MvNormal([0.0, 0.0], 4.0*diagm(ones(2))), x)
            model = PSDModel(Legendre(-10.0..10.0)^2, :downward_closed, 5)
            # generate some data
            X = [(rand(2) * 20.0 .- 10.0) for i in 1:500]
            Y = f.(X)
            α_divergence_fit!(model, 2.0, X, Y; trace=false, use_putinar=false)
            normalize!(model)

            @test mean((model.(X) .- Y).^2) < 1e-3
        end
        @testset "α=10" begin
            f(x) = pdf(MvNormal([0.0, 0.0], 4.0*diagm(ones(2))), x)
            model = PSDModel(Legendre(-10.0..10.0)^2, :downward_closed, 5)
            # generate some data
            X = [(rand(2) * 20.0 .- 10.0) for i in 1:500]
            Y = f.(X)
            α_divergence_fit!(model, 10.0, X, Y, trace=false, use_putinar=false)
            normalize!(model)

            @test mean((model.(X) .- Y).^2) < 1e-3
        end
        @testset "α=-1" begin
            f(x) = pdf(MvNormal([0.0, 0.0], 4.0*diagm(ones(2))), x)
            model = PSDModel(Legendre(-10.0..10.0)^2, :downward_closed, 5)
            # generate some data
            X = [(rand(2) * 20.0 .- 10.0) for i in 1:500]
            Y = f.(X)
            α_divergence_fit!(model, -1.0, X, Y, trace=false, use_putinar=false)
            normalize!(model)

            @test mean((model.(X) .- Y).^2) < 1e-3
        end
    end
end