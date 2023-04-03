using PSDModels: FMTensorPolynomial, reduce_dim, trivial_TensorPolynomial, dimensions
using FastGaussQuadrature: gausslegendre, gausschebyshev, gausshermite
using ApproxFun

@testset "Orthogonality check default domain" begin
    @testset "Chebyshev orthonormality" begin
        poly = trivial_TensorPolynomial(Chebyshev(), 10)

        # test dimensions
        @test dimensions(poly) == 1
        x, w = gausschebyshev(20)

        # normality check
        for i=1:10
            eval_poly = [poly(xi)[i] for xi in x]
            res = sum(w .* eval_poly.^2)
            @test res ≈ 1.0
        end

        # orthogonality check
        for i=1:10, j=1:10
            if i != j
                eval_poly = [poly(xi)[i] for xi in x]
                eval_poly2 = [poly(xi)[j] for xi in x]
                res = sum(w .* eval_poly .* eval_poly2)
                @test isapprox(res, 0.0, atol=1e-10)
            end
        end
    end

    @testset "Legendre orthonormality" begin
        poly = trivial_TensorPolynomial(Legendre(), 10)

        x, w = gausslegendre(20)

        # normality check
        for i=1:10
            eval_poly = [poly(xi)[i] for xi in x]
            res = sum(w .* eval_poly.^2)
            @test res ≈ 1.0
        end

        # orthogonality check
        for i=1:10, j=1:10
            if i != j
                eval_poly = [poly(xi)[i] for xi in x]
                eval_poly2 = [poly(xi)[j] for xi in x]
                res = sum(w .* eval_poly .* eval_poly2)
                @test isapprox(res, 0.0, atol=1e-10)
            end
        end
    end

    @testset "Hermite orthonormality" begin
        poly = trivial_TensorPolynomial(Hermite(), 10)

        x, w = gausshermite(20)

        # normality check
        for i=1:10
            eval_poly = [poly(xi)[i] for xi in x]
            res = sum(w .* eval_poly.^2)
            @test res ≈ 1.0
        end

        # orthogonality check
        for i=1:10, j=1:10
            if i != j
                eval_poly = [poly(xi)[i] for xi in x]
                eval_poly2 = [poly(xi)[j] for xi in x]
                res = sum(w .* eval_poly .* eval_poly2)
                @test isapprox(res, 0.0, atol=1e-10)
            end
        end
    end
end


@testset "Orthogonality check scaled domain" begin
    @testset "Chebyshev orthonormality" begin
        l = 0.0
        r = 10.0
        poly = trivial_TensorPolynomial(Chebyshev(l..r), 10)

        # test dimensions
        @test dimensions(poly) == 1
        x, w = gausschebyshev(20)
        x .*= ((r - l)/2)
        x .+= ((r + l)/2)
        scale = ((r - l)/2)

        # normality check
        for i=1:10
            eval_poly = [poly(xi)[i] for xi in x]
            res = scale * sum(w .* eval_poly.^2)
            @test res ≈ 1.0
        end

        # orthogonality check
        for i=1:10, j=1:10
            if i != j
                eval_poly = [poly(xi)[i] for xi in x]
                eval_poly2 = [poly(xi)[j] for xi in x]
                res = scale * sum(w .* eval_poly .* eval_poly2)
                @test isapprox(res, 0.0, atol=1e-10)
            end
        end
    end

    @testset "Legendre orthonormality" begin
        l = 0.0
        r = 10.0
        poly = trivial_TensorPolynomial(Legendre(l..r), 10)

        x, w = gausslegendre(20)
        x .*= ((r - l)/2)
        x .+= ((r + l)/2)
        scale = ((r - l)/2)

        # normality check
        for i=1:10
            eval_poly = [poly(xi)[i] for xi in x]
            res = scale * sum(w .* eval_poly.^2)
            @test res ≈ 1.0
        end

        # orthogonality check
        for i=1:10, j=1:10
            if i != j
                eval_poly = [poly(xi)[i] for xi in x]
                eval_poly2 = [poly(xi)[j] for xi in x]
                res = scale * sum(w .* eval_poly .* eval_poly2)
                @test isapprox(res, 0.0, atol=1e-10)
            end
        end
    end
end