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
            ref_map = SMT.AlgebraicReference{d, Float64}()
            f_app = SMT.pushforward(ref_map, x->pdf(MvNormal(zeros(d), 1.0), x))
            rng = 0.001:0.001:0.999
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
    @testset "exp(log(T^♯ f)) normalized" begin
        for d=1:2
            ref_map = SMT.AlgebraicReference{d, Float64}()
            f_app = SMT.log_pushforward(ref_map, x->logpdf(MvNormal(zeros(d), 1.0), x))
            f_app = let f_app=f_app 
                x->exp(f_app(x))
            end
            rng = 0.001:0.001:0.999
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
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
            f_app = SMT.pushforward(ref_map, x->pdf(MvNormal(zeros(d), 1.0), x))
            rng = 0.001:0.001:0.999
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-2)
        end
    end
    @testset "exp(log(T^♯ f)) normalized" begin
        for d=1:2
            ref_map = SMT.AlgebraicReference{d, Float64}()
            f_app = SMT.log_pushforward(ref_map, x->logpdf(MvNormal(zeros(d), 1.0), x))
            f_app = let f_app=f_app 
                x->exp(f_app(x))
            end
            rng = 0.001:0.001:0.999
            iter = Iterators.product(fill(rng, d)...)
            @test isapprox(0.001^d*sum(f_app([x...]) for x in iter), 1.0, atol=1e-3)
        end
    end
end