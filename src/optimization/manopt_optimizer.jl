import Manopt
import Manifolds
import ManifoldDiff


function _λ1_regularization(A::AbstractMatrix{T}) where {T<:Number}
    return tr(A)
end

function _λ1_regularization_gradient!(grad_A::AbstractMatrix{T}, A::AbstractMatrix{T}, λ_1::T) where {T<:Number}
    grad_A .= grad_A + λ_1 * I
    return nothing
end

function _λ2_regularization(A::AbstractMatrix{T}) where {T<:Number}
    return sum(A.^2)
end

function _λ2_regularization_gradient!(grad_A::AbstractMatrix{T}, A::AbstractMatrix{T}, λ_2::T) where {T<:Number}
    grad_A .+= λ_2 * 2.0 * A
    return nothing
end

function _λ_id_regularization(a::PSDModel{T}, A::AbstractMatrix{T}, λ_id::T, N_id::Int) where {T<:Number}
    return λ_id * sum((a(x, A) - one(T))^2 for x in eachcol(rand(d, N_id)))
end

function _λ_id_regularization_gradient!(grad_A::AbstractMatrix{T}, a::PSDModel{T}, A::AbstractMatrix{T}, λ_id::T, N_id::Int) where {T<:Number}
    _helper(x) = begin
        v = Φ(a, x)
        return (dot(v, A, v) - one(T)) * v * v'
    end
    grad_A .+= 2.0 * λ_id * sum(_helper(x) for x in eachcol(rand(d, N_id)))
    return nothing
end

abstract type _grad_struct end

struct _grad_cost_alpha <: _grad_struct
    model
    α
    grad_A
    X
    Y
    λ_1
    λ_2
end


function (a::_grad_cost_alpha)(_, A)
    @inline _help(x) = begin
        v = Φ(a.model, x)
        ret = dot(v, A, v)^(-a.α) * (v * v')
        return ret
    end
    # put grad_A to zero
    a.grad_A .= 0.0
    for (x, y) in zip(a.X, a.Y)
        res = _help(x) * y^(a.α)
        # @show res
        a.grad_A .+= res
    end
    # @show a.grad_A
    a.grad_A .*= (1/length(a.Y)) * (-1/a.α)
    a.grad_A .= a.grad_A + (1/ a.α) * diagm(0=>ones(size(A, 1)))
    _λ1_regularization_gradient!(a.grad_A, A, a.λ_1)
    _λ2_regularization_gradient!(a.grad_A, A, a.λ_2)
    return nothing
end

struct _grad_ML <: _grad_struct
    model
    grad_A
    X
    λ_1
    λ_2
end

function (a::_grad_ML)(_, A)
    @inline _help(x) = begin
        v = Φ(a.model, x)
        ret = (1/dot(v, A, v)) * (v * v')
        return ret
    end
    # put grad_A to zero
    a.grad_A .= 0.0
    for x in a.X
        a.grad_A .+= _help(x)
    end
    a.grad_A .*= -(1/length(a.X))
    a.grad_A .= a.grad_A + diagm(0=>ones(size(A, 1)))
    _λ1_regularization_gradient!(a.grad_A, A, a.λ_1)
    _λ2_regularization_gradient!(a.grad_A, A, a.λ_2)
    return nothing
end

struct _grad_KL <: _grad_struct
    model
    grad_A
    X
    Y
    λ_1
    λ_2
end

function (a::_grad_KL)(_, A)
    @inline _help(x) = begin
        v = Φ(a.model, x)
        ret = (1/dot(v, A, v)) * (v * v')
        return ret
    end
    # put grad_A to zero
    a.grad_A .= 0.0
    for (x, y) in zip(a.X, a.Y)
        a.grad_A .+= _help(x) * y
    end
    a.grad_A .*= -(1/length(a.X))
    a.grad_A .= a.grad_A + I
    _λ1_regularization_gradient!(a.grad_A, A, a.λ_1)
    _λ2_regularization_gradient!(a.grad_A, A, a.λ_2)
    return nothing
end


struct ManoptOptPropblem
    M::Manifolds.AbstractManifold       # Manifold and metric used, by default standard SDP with affine metric
    cost_func                           # cost function
    grad_cost!::_grad_struct        # struct of gradient of cost function together with field grad_A for euclidean gradient
    grad_cost_M!                        # gradient of cost function on Manifold, only in place version
    algorithm::Symbol                   # algorithm used, by default Riemannian steepest descent
    function ManoptOptPropblem(M::Manifolds.AbstractManifold, cost_func, grad_cost!::_grad_struct)
        new(M, cost_func, grad_cost!, nothing, :gradient_descent)
    end
    function ManoptOptPropblem(M::Manifolds.AbstractManifold, cost_func, grad_cost!::_grad_struct, algorithm::Symbol)
        new(M, cost_func, grad_cost!, nothing, algorithm)
    end
    function ManoptOptPropblem(M::Manifolds.AbstractManifold, cost_func, grad_cost!, grad_cost_M!)
        new(M, cost_func, grad_cost!, grad_cost_M!, :gradient_descent)
    end
end

function ManoptOptPropblem(cost_func, grad_cost!::_grad_struct, N::Int; algorithm=:gradient_descent)
    M = Manifolds.SymmetricPositiveDefinite(N)
    ManoptOptPropblem(M, cost_func, grad_cost!, algorithm)
end

function _grad_cost_M!(prob::ManoptOptPropblem, M, grad_A, A)
    prob.grad_cost!(grad_A, A)
    ManifoldDiff.riemannian_gradient!(M, grad_A, A, prob.grad_cost!.grad_A)
    return nothing
end

function optimize(prob::ManoptOptPropblem, A_init;
            maxit=1000, trace=false,
            stepsize=nothing)
    M = prob.M
    cost_func = prob.cost_func
    grad_cost_M! = if prob.grad_cost_M! === nothing
        let prob=prob
            (M, grad_A, A) -> _grad_cost_M!(prob, M, grad_A, A)
        end
    else
        prob.grad_cost_M!
    end

    debug = if trace
            [:Iteration,(:Change, "|Δp|: %1.9f |"), 
            (:Cost, " F(x): %1.11f | "), 
            (:Stepsize, " s: %f | "), 5, "\n", :Stop]
    else
        []
    end
    
    stopping_criterion = (Manopt.StopAfterIteration(maxit) |Manopt.StopWhenGradientNormLess(1e-8))

    _stepsize = if stepsize === nothing
        # Manopt.PolyakStepsize(i->1/i, 0.0)
        Manopt.default_stepsize(M, Manopt.GradientDescentState)
    else
        stepsize
    end

    A_sol = Matrix(deepcopy(A_init))
    if prob.algorithm == :gradient_descent
        Manopt.gradient_descent!(M, cost_func, grad_cost_M!, A_sol, 
                evaluation=Manopt.InplaceEvaluation(),
                stopping_criterion=stopping_criterion,
                debug=debug,
                stepsize=_stepsize)
    elseif prob.algorithm == :stochastic_gradient_descent
        Manopt.stochastic_gradient_descent!(M, grad_cost_M!, A_sol,
                cost=cost_func,
                evaluation=Manopt.InplaceEvaluation(),
                debug=debug)
    else
        throw(error("Algorithm $(prob.algorithm) not implemented."))
        return nothing
    end
    return Hermitian(A_sol)
end


function _α_divergence_Manopt!(a::PSDModel{T},
                α::T,
                X::PSDDataVector{T},
                Y::AbstractVector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                normalization=false,
                algorithm=:gradient_descent,
                kwargs...
            ) where {T<:Number}

    @assert normalization == false "Normalization not implemented yet."
    @assert α != 1 "Use KL divergence instead"
    @assert α != 0 "Use reversed KL divergence instead"

    function _cost_alpha(A, α, X, Y)
        res = zero(T)
        for (x, y) in zip(X, Y)
            v = Φ(a, x)
            res += dot(v, A, v)^(1-α) * y^(α)
        end
        y_int = (1/length(Y)) * (1/(α-1)) * sum(Y)
        res = (1/length(Y)) * (1/(α * (α - 1))) * res + (1/α)* tr(A) - y_int
        res += λ_1 * _λ1_regularization(A) + λ_2 * _λ2_regularization(A)
    end

    cost_alpha = let α=α, X=X, Y=Y
        (M, A) -> _cost_alpha(A, α, X, Y)
    end

    N = size(a.B, 1)

    grad_alpha! = _grad_cost_alpha(a, α, zeros(T, N, N), X, Y, λ_1, λ_2)

    prob = ManoptOptPropblem(cost_alpha, grad_alpha!, N, algorithm=algorithm)
    A_new = optimize(prob, a.B; trace=trace, kwargs...)
    set_coefficients!(a, A_new)
    return cost_alpha(nothing, A_new)
end



function _ML_Manopt!(a::PSDModel{T},
                X::PSDDataVector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                normalization=false,
                algorithm=:gradient_descent,
                kwargs...
            ) where {T<:Number}

    @assert normalization == false "Normalization not implemented yet."

    function _cost_ML(A, X)
        res = zero(T)
        for x in X
            # v = Φ(a, x)
            res += -log(a(x, A))
        end
        res = (1/length(X)) * res + tr(A)
        res += λ_1 * _λ1_regularization(A) + λ_2 * _λ2_regularization(A)
        return res
    end

    cost_ML = let X=X
        (M, A) -> _cost_ML(A, X)
    end

    N = size(a.B, 1)

    grad_ML! = _grad_ML(a, zeros(T, N, N), X, λ_1, λ_2)

    prob = ManoptOptPropblem(cost_ML, grad_ML!, N; algorithm=algorithm)
    A_new = optimize(prob, a.B; trace=trace, kwargs...)
    A_new = A_new / tr(A_new)
    set_coefficients!(a, A_new)
    return cost_ML(nothing, A_new)
end


function _KL_Manopt!(a::PSDModel{T},
                X::PSDDataVector{T},
                Y::AbstractVector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                normalization=false,
                algorithm=:gradient_descent,
                kwargs...
            ) where {T<:Number}

    @assert normalization == false "Normalization not implemented yet."

    function _cost_KL(A, X, Y)
        res = zero(T)
        for (x, y) in zip(X, Y)
            res += (log(y)-log(a(x, A))-1) * y
        end
        res = (1/length(X)) * res + tr(A)
        res += λ_1 * _λ1_regularization(A) + λ_2 * _λ2_regularization(A)
        return res
    end

    cost_KL = let X=X
        (M, A) -> _cost_KL(A, X, Y)
    end

    N = size(a.B, 1)

    grad_KL! = _grad_KL(a, zeros(T, N, N), X, Y, λ_1, λ_2)

    prob = ManoptOptPropblem(cost_KL, grad_KL!, N; algorithm=algorithm)
    A_new = optimize(prob, a.B; trace=trace, kwargs...)
    # A_new = A_new / tr(A_new)
    set_coefficients!(a, A_new)
    return cost_KL(nothing, A_new)
end