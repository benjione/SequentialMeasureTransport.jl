import SequentialMeasureTransport as SMT

@testset "Scaling Reference" begin
    @testset "T^{-1}(T(x)) = x" begin
        for d=1:5
            L = rand(d)
            R = L + 2.0 * rand(d) .+ 0.1
            ref_map = SMT.ScalingReference{d}(L, R)

            for i=1:100
                x = rand(d)
                @test SMT.pullback(ref_map, SMT.pushforward(ref_map, x)) ≈ x
            end
        end
    end
    @testset "T^♯ T_♯ f = f" begin
        for d=1:5
            L = rand(d)
            R = L + 2.0 * rand(d) .+ 0.1
            ref_map = SMT.ScalingReference{d}(L, R)
            f_app = SMT.pullback(ref_map, x->1.0)
            f_app_2 = SMT.pushforward(ref_map, f_app)

            for i=1:100
                x = rand(d)
                @test f_app_2(x) ≈ 1.0
            end
        end
    end
    @testset "log(T^♯ T_♯ f) = log(f)" begin
        for d=1:5
            L = rand(d)
            R = L + 2.0 * rand(d) .+ 0.1
            ref_map = SMT.ScalingReference{d}(L, R)
            f_app = SMT.log_pullback(ref_map, x->log(1.0))
            f_app_2 = SMT.log_pushforward(ref_map, f_app)

            for i=1:100
                x = rand(d)
                @test exp(f_app_2(x)) ≈ 1.0
            end
        end
    end
    @testset "T^♯ f normalized" begin
        for d=1:2
            L = rand(d)
            R = L + 2.0 * rand(d) .+ 0.1
            V = prod(R - L)
            ref_map = SMT.ScalingReference{d}(L, R)
            f_app = SMT.pushforward(ref_map, x->1/V)
            rng = 0.001:0.001:0.999
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
    @testset "exp( log(T^♯ f)) normalized" begin
        for d=1:2
            L = rand(d)
            R = L + 2.0 * rand(d) .+ 0.1
            V = prod(R - L)
            ref_map = SMT.ScalingReference{d}(L, R)
            f_app = SMT.log_pushforward(ref_map, x->log(1/V))
            f_app = let f_app=f_app 
                x->exp(f_app(x))
            end
            rng = 0.001:0.001:0.999
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
    @testset "marginal T^♯ f normalized" begin
        for d=2:3
            L = rand(d)
            R = L + 2.0 * rand(d) .+ 0.1
            V = prod(R - L)
            V_marg = prod(R[1:d-1] - L[1:d-1])
            ref_map = SMT.ScalingReference{d, 1}(L, R)
            f_app = SMT.marginal_pushforward(ref_map, x->1/V_marg)
            rng = 1e-9:0.001:1.0-1e-9
            iter = Iterators.product(fill(rng, d-1)...)
            @test isapprox(0.001^(d-1)*sum(f_app([x...]) for x in iter), 1.0, atol=1e-1)
        end
    end
end

@testset "Gaussian Reference" begin
    @testset "T^{-1}(T(x)) = x" begin
        for d=1:5
            ref_map = SMT.GaussianReference{d, Float64}()

            for i=1:100
                x = rand(d)
                @test SMT.pullback(ref_map, SMT.pushforward(ref_map, x)) ≈ x
            end
        end
    end
    @testset "T^♯ T_♯ f = f" begin
        for d=1:5
            for _=1:5
                s = rand() * 10.0
                ref_map = SMT.GaussianReference{d, Float64}(s)
                f_app = SMT.pullback(ref_map, x->1.0)
                f_app_2 = SMT.pushforward(ref_map, f_app)
        
                for i=1:100
                    x = rand(d)
                    @test f_app_2(x) ≈ 1.0
                end
            end
        end
    end
    @testset "log(T^♯ T_♯ f) = lof(f)" begin
        for d=1:5
            for _=1:5
                s = rand() * 10.0
                ref_map = SMT.GaussianReference{d, Float64}(s)
                f_app = SMT.log_pullback(ref_map, x->0.0)
                f_app_2 = SMT.log_pushforward(ref_map, f_app)
        
                for i=1:100
                    x = rand(d)
                    @test f_app_2(x) ≈ 0.0
                end
            end
        end
    end
    @testset "T^♯ f normalized" begin
        for d=1:2
            ref_map = SMT.GaussianReference{d, Float64}()
            f_app = SMT.pushforward(ref_map, x->pdf(MvNormal(zeros(d), I), x))
            rng = 0.001:0.001:0.999
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
    @testset "exp(log(T^♯ f)) normalized" begin
        for d=1:2
            ref_map = SMT.GaussianReference{d, Float64}()
            f_app = SMT.log_pushforward(ref_map, x->logpdf(MvNormal(zeros(d), I), x))
            f_app = let f_app=f_app 
                x->exp(f_app(x))
            end
            rng = 0.001:0.001:0.999
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
    @testset "marginal T^♯ f normalized" begin
        for d=2:3
            ref_map = SMT.GaussianReference{d, 1, Float64}()
            f_app = SMT.marginal_pushforward(ref_map, x->pdf(MvNormal(ones(d-1), diagm(ones(d-1))), x))
            rng = 1e-9:0.0005:1.0-1e-9
            iter = Iterators.product(fill(rng, d-1)...)
            @test isapprox(0.0005^(d-1)*sum(f_app([x...]) for x in iter), 1.0, atol=1e-1)
        end
    end
end

@testset "Algebraic Reference" begin
    @testset "T^{-1}(T(x)) = x" begin
        for d=1:5
            ref_map = SMT.AlgebraicReference{d, Float64}()

            for i=1:100
                x = rand(d)
                @test SMT.pullback(ref_map, SMT.pushforward(ref_map, x)) ≈ x
            end
        end
    end
    @testset "T^♯ T_♯ f = f" begin
        for d=1:5
            ref_map = SMT.AlgebraicReference{d, Float64}()
            f_app = SMT.pullback(ref_map, x->1.0)
            f_app_2 = SMT.pushforward(ref_map, f_app)
    
            for i=1:100
                x = rand(d)
                @test f_app_2(x) ≈ 1.0
            end
        end
    end
    @testset "log(T^♯ T_♯ f) = log(f)" begin
        for d=1:5
            ref_map = SMT.AlgebraicReference{d, Float64}()
            f_app = SMT.log_pullback(ref_map, x->0.0)
            f_app_2 = SMT.log_pushforward(ref_map, f_app)
    
            for i=1:100
                x = rand(d)
                @test isapprox(f_app_2(x), 0.0, atol=1e-13)
            end
        end
    end
    @testset "T^♯ f normalized" begin
        for d=1:2
            ref_map = SMT.AlgebraicReference{d, Float64}()
            f_app = SMT.pushforward(ref_map, x->pdf(MvNormal(zeros(d), I), x))
            rng = 0.001:0.001:0.999
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
    @testset "exp(log(T^♯ f)) normalized" begin
        for d=1:2
            ref_map = SMT.AlgebraicReference{d, Float64}()
            f_app = SMT.log_pushforward(ref_map, x->logpdf(MvNormal(zeros(d), I), x))
            f_app = let f_app=f_app 
                x->exp(f_app(x))
            end
            rng = 0.001:0.001:0.999
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-3)
        end
    end
    @testset "marginal T^♯ f normalized" begin
        for d=2:3
            ref_map = SMT.AlgebraicReference{d, 1, Float64}()
            f_app = SMT.marginal_pushforward(ref_map, x->pdf(MvNormal(ones(d-1), diagm(ones(d-1))), x))
            rng = 1e-8:0.001:1.0-1e-8
            iter = Iterators.product(fill(rng, d-1)...)
            @test isapprox(0.001^(d-1)*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
end


@testset "Composed Reference: normalized Gaussian" begin
    @testset "T^{-1}(T(x)) = x" begin
        for d=1:5
            means, stds = randn(d), 2.0*rand(d)
            ref_map = SMT.normalized_gaussian_reference(means, stds)

            for i=1:100
                x = rand(MvNormal(means, stds))
                @test SMT.pullback(ref_map, SMT.pushforward(ref_map, x)) ≈ x
            end
        end
    end
    @testset "T^♯ T_♯ f = f" begin
        for d=1:5
            ref_map = SMT.normalized_gaussian_reference(randn(d), 2.0*rand(d))
            f_app = SMT.pullback(ref_map, x->1.0)
            f_app_2 = SMT.pushforward(ref_map, f_app)
    
            for i=1:100
                x = rand(d)
                @test f_app_2(x) ≈ 1.0
            end
        end
    end
    @testset "log(T^♯ T_♯ f) = log(f)" begin
        for d=1:5
            ref_map = SMT.normalized_gaussian_reference(randn(d), 2.0*rand(d))
            f_app = SMT.log_pullback(ref_map, x->0.0)
            f_app_2 = SMT.log_pushforward(ref_map, f_app)
    
            for i=1:100
                x = rand(d)
                @test isapprox(f_app_2(x), 0.0, atol=1e-13)
            end
        end
    end
    @testset "T^♯ f normalized" begin
        for d=1:2
            means = randn(d)
            stds = 0.1*ones(d) + rand(d)
            ref_map = SMT.normalized_gaussian_reference(means, stds)
            f_app = SMT.pushforward(ref_map, x->pdf(MvNormal(means, diagm(stds.^2)), x))
            rng = 1e-8:0.001:1.0-1e-8
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
    @testset "exp(log(T^♯ f)) normalized" begin
        for d=1:2
            means = randn(d)
            stds = 0.1*ones(d) + rand(d)
            ref_map = SMT.normalized_gaussian_reference(means, stds)
            # f_app = SMT.pushforward(ref_map, x->pdf(MvNormal(means, diagm(stds)), x))
            f_app = SMT.log_pushforward(ref_map, x->logpdf(MvNormal(means, diagm(stds.^2)), x))
            f_app = let f_app=f_app 
                x->exp(f_app(x))
            end
            rng = 1e-8:0.001:1.0-1e-8
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-3)
        end
    end
    @testset "marginal T^♯ f normalized" begin
        for d=2:3
            means = randn(d)
            stds = 0.1*ones(d) + rand(d)
            ref_map = SMT.normalized_gaussian_reference(means, stds, 1)
            # f_app = SMT.marginal_pushforward(ref_map, x->pdf(MvNormal(means[1:d-1], diagm(stds[1:d-1].^2)), x))
            f_app_1 = SMT.marginal_pushforward(ref_map.components[2], x->pdf(MvNormal(means[1:d-1], diagm(stds[1:d-1].^2)), x))
            f_app = SMT.marginal_pushforward(ref_map.components[1], x->f_app_1(x))
            rng = 1e-8:0.001:1.0-1e-8
            iter = Iterators.product(fill(rng, d-1)...)
            @test isapprox(0.001^(d-1)*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
end


@testset "Composed Reference: normalized Algebraic" begin
    @testset "T^{-1}(T(x)) = x" begin
        for d=1:5
            means, stds = randn(d), 2.0*rand(d)
            ref_map = SMT.normalized_algebraic_reference(means, stds)

            for i=1:100
                x = rand(MvNormal(means, stds))
                @test SMT.pullback(ref_map, SMT.pushforward(ref_map, x)) ≈ x
            end
        end
    end
    @testset "T^♯ T_♯ f = f" begin
        for d=1:5
            ref_map = SMT.normalized_algebraic_reference(randn(d), 2.0*rand(d))
            f_app = SMT.pullback(ref_map, x->1.0)
            f_app_2 = SMT.pushforward(ref_map, f_app)
    
            for i=1:100
                x = rand(d)
                @test f_app_2(x) ≈ 1.0
            end
        end
    end
    @testset "log(T^♯ T_♯ f) = log(f)" begin
        for d=1:5
            ref_map = SMT.normalized_algebraic_reference(randn(d), 2.0*rand(d))
            f_app = SMT.log_pullback(ref_map, x->0.0)
            f_app_2 = SMT.log_pushforward(ref_map, f_app)
    
            for i=1:100
                x = rand(d)
                @test isapprox(f_app_2(x), 0.0, atol=1e-13)
            end
        end
    end
    @testset "T^♯ f normalized" begin
        for d=1:2
            means = randn(d)
            stds = 0.1*ones(d) + rand(d)
            ref_map = SMT.normalized_algebraic_reference(means, stds)
            f_app = SMT.pushforward(ref_map, x->pdf(MvNormal(means, diagm(stds.^2)), x))
            rng = 1e-8:0.001:1.0-1e-8
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
    @testset "T^♯ T_♯ ρ normalized" begin
        for d=1:2
            means = randn(d)
            stds = 0.1*ones(d) + rand(d)
            ref_map = SMT.normalized_algebraic_reference(means, stds)
            f_app = SMT.pullback(ref_map, x->1.0)
            f_app_2 = SMT.pushforward(ref_map, f_app)
            rng = 1e-8:0.001:1.0-1e-8
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app_2([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
    @testset "exp(log(T^♯ f)) normalized" begin
        for d=1:2
            means = randn(d)
            stds = 0.1*ones(d) + rand(d)
            ref_map = SMT.normalized_algebraic_reference(means, stds)
            # f_app = SMT.pushforward(ref_map, x->pdf(MvNormal(means, diagm(stds)), x))
            f_app = SMT.log_pushforward(ref_map, x->logpdf(MvNormal(means, diagm(stds.^2)), x))
            f_app = let f_app=f_app 
                x->exp(f_app(x))
            end
            rng = 1e-8:0.001:1.0-1e-8
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-3)
        end
    end
    @testset "marginal T^♯ f normalized" begin
        for d=2:3
            means = randn(d)
            stds = 0.1*ones(d) + rand(d)
            ref_map = SMT.normalized_algebraic_reference(means, stds, 1)
            f_app = SMT.marginal_pushforward(ref_map, x->pdf(MvNormal(means[1:d-1], diagm(stds[1:d-1].^2)), x))
            rng = 1e-8:0.001:1.0-1e-8
            iter = Iterators.product(fill(rng, d-1)...)
            @test isapprox(0.001^(d-1)*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
end