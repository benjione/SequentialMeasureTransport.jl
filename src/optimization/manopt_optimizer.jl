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

mutable struct CrossValidationStopping{T <: Number} <: Manopt.StoppingCriterion
    CV_X :: PSDDataVector{T}
    CV_Y :: AbstractVector{T}
    loss :: Function
    CV_seq :: AbstractVector{T}
    function CrossValidationStopping(CV_X::PSDDataVector{T}, 
                    CV_Y::AbstractVector{T},
                    loss::Function) where {T<:Number}
        new{T}(CV_X, CV_Y, loss, T[])
    end
end
function (c::CrossValidationStopping{T})(::Manopt.AbstractManoptProblem, 
                            state::Manopt.AbstractManoptSolverState, 
                            i::Int) where {T<:Number}
    A = Manopt.get_iterate(state)
    new_cv_loss = c.loss(c.CV_X, c.CV_Y, A)
    push!(c.CV_seq, new_cv_loss)
    if length(c.CV_seq) < 5
        return false
    end
    if c.CV_seq[end] > c.CV_seq[end-1]
        return true
    end 
    return false
end
function Manopt.get_reason(c::CrossValidationStopping)
    return "Stopping by CV with CV loss of $(c.CV_seq[end]).\n"
end
function Manopt.status_summary(c::CrossValidationStopping)
    return "Cross validation reached $(c.CV_seq[end])."
end
Manopt.indicates_convergence(c::CrossValidationStopping) = true
function show(io::IO, c::CrossValidationStopping)
    return print(io, "StopAfter( $(status_summary(c)) )" )
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
        a.grad_A .+= res
    end
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
            mingrad_stop=1e-8,
            custom_stopping_criterion=nothing,
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
    
    stopping_criterion = if custom_stopping_criterion === nothing
        (Manopt.StopAfterIteration(maxit) |Manopt.StopWhenGradientNormLess(mingrad_stop))
    else
        custom_stopping_criterion
    end

    _stepsize = if stepsize === nothing
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

include("../extra/adaptive_sampling/stopping_rule_MC_sampling.jl")

function _adaptive_α_divergence_Manopt!(a::PSDModelOrthonormal{d, T},
                α::T,
                X::PSDDataVector{T},
                Y::AbstractVector{T},
                g::Function, δ::T, p::T;
                λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                normalization=false,
                algorithm=:gradient_descent,
                N0=200,
                Nmax=100000,
                maxit=1000,
                adaptive_sample_steps=10,
                addmax=500,
                addmin=10,
                rand_gen=nothing,
                kwargs...
            ) where {d, T<:Number}

    @assert normalization == false "Normalization not implemented yet."
    @assert α != 1 "Use KL divergence instead"
    @assert α != 0 "Use reversed KL divergence instead"

    @assert δ > 0 "δ must be positive"
    @assert 0 < p < 1 "p must be in (0, 1)"

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

    N_sample = length(Y)
    for i in 1:adaptive_sample_steps
        _rand_gen = if rand_gen === nothing
            () -> rand(T, d)
        else
            rand_gen
        end
        loss = (x, y) -> begin
            # res = zero(T)
            v = Φ(a, x)
            res = dot(v, a.B, v)^(1-α) * y^(α)
            y_int = (1/(α-1)) * y
            res = (1/(α * (α - 1))) * res + (1/α)* res - y_int
            # res += λ_1 * _λ1_regularization(A) + λ_2 * _λ2_regularization(A)
            return res
        end
        # do not sample adaptive in first round, variance too unreliable
        
        sample_adaptive!(g, loss, X, Y, _rand_gen, _chebyshev_stopping_rule, δ, p; 
                    N0=N0, Nmax=Nmax, addmax=addmax, addmin=addmin)
    

        trace && println("Iteration $i: $(length(Y)) samples, added $(length(Y) - N_sample) samples.")
        N_sample = length(Y)

        cost_alpha = let α=α, X=X, Y=Y
            (M, A) -> _cost_alpha(A, α, X, Y)
        end

        N = size(a.B, 1)

        grad_alpha! = _grad_cost_alpha(a, α, zeros(T, N, N), X, Y, λ_1, λ_2)

        prob = ManoptOptPropblem(cost_alpha, grad_alpha!, N, algorithm=algorithm)
        A_new = optimize(prob, a.B; trace=trace, maxit=Int(floor(maxit/adaptive_sample_steps)), kwargs...)
        set_coefficients!(a, A_new)
    end
    return _cost_alpha(a.B, α, X, Y)
end


function _adaptive_CV_α_divergence_Manopt!(a::PSDModelOrthonormal{d, T},
                α::T,
                X::PSDDataVector{T},
                Y::AbstractVector{T},
                g::Function;
                λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                normalization=false,
                algorithm=:gradient_descent,
                N0=500,
                Nadd_per_iter=200,
                adaptive_sample_steps=10,
                CV_split=0.8,
                broadcasted_target = false,
                rand_gen=nothing,
                maxit=1000,
                kwargs...
            ) where {d, T<:Number}

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
    _rand_gen = if rand_gen === nothing
        () -> rand(T, d)
    else
        rand_gen
    end
    ## create N0 samples
    function add_samples(N) 
        for _ = 1:N
            x = _rand_gen()
            push!(X, x)
        end
        _Y = if broadcasted_target
            g(X[end-N+1:end])
        else
            [g(x) for x in X[end-N+1:end]]
        end
        append!(Y, _Y)
    end
    if length(Y) < N0
        add_samples(N0 - length(Y))
    end
    CV_loss(X, Y, A) = _cost_alpha(A, α, X, Y)

    for i in 1:adaptive_sample_steps
        if i ≠ 1
            add_samples(Nadd_per_iter)
        end
        # split into train and test, with 10 percent test data
        shuf = Random.shuffle(1:length(X))
        _X, _Y = X[shuf], Y[shuf]
        X_train = _X[1:round(Int, CV_split * length(_X))]
        Y_train = _Y[1:round(Int, CV_split * length(_Y))]
        X_test = _X[round(Int, CV_split * length(_X))+1:end]
        Y_test = _Y[round(Int, CV_split * length(_Y))+1:end]

        cost_alpha = let α=α, X=X_train, Y=Y_train
            (M, A) -> _cost_alpha(A, α, X, Y)
        end

        N = size(a.B, 1)

        stop_CV = CrossValidationStopping(X_test, Y_test, CV_loss) | Manopt.StopAfterIteration(maxit)

        grad_alpha! = _grad_cost_alpha(a, α, zeros(T, N, N), 
                        X_train, Y_train, λ_1, λ_2)

        prob = ManoptOptPropblem(cost_alpha, grad_alpha!, N, algorithm=algorithm)
        A_new = optimize(prob, a.B; trace=trace, 
                custom_stopping_criterion=stop_CV, kwargs...)
        set_coefficients!(a, A_new)
    end
    return _cost_alpha(a.B, α, X, Y)
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