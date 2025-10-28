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
    ϵ_CV :: T   # stopping criterion if CV loss increases by ϵ_CV
    coef_list 
    D_list 
    function CrossValidationStopping(CV_X::PSDDataVector{T}, 
                    CV_Y::AbstractVector{T},
                    loss::Function; ϵ_CV=1e-3, coef_list=nothing, D_list=nothing) where {T<:Number}
        new{T}(CV_X, CV_Y, loss, T[], ϵ_CV, coef_list, D_list)
    end
end
function (c::CrossValidationStopping{T})(prob::Manopt.AbstractManoptProblem, 
                            state::Manopt.AbstractManoptSolverState, 
                            i::Int) where {T<:Number}
    A = Manopt.get_iterate(state)
    if c.coef_list !== nothing && c.D_list !== nothing
        A = _p_to_A(Manopt.get_manifold(prob), A, c.D_list, c.coef_list)
    end
    new_cv_loss = c.loss(c.CV_X, c.CV_Y, A)
    push!(c.CV_seq, new_cv_loss)
    if length(c.CV_seq) < 5
        return false
    end
    if c.CV_seq[end] > (1 + c.ϵ_CV) * minimum(c.CV_seq)
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

add_grad!(a::AbstractArray{T}, b::AbstractArray{T}) where {T<:Number} = a .+= b

struct _grad_cost_alpha <: _grad_struct
    model
    α
    A
    M
    grad_A
    grad_p
    X
    Y
    λ_1
    λ_2
    ϵ
    threading::Bool
    function _grad_cost_alpha(model, α, A, M, grad_A, grad_p, X, Y, λ_1, λ_2; ϵ=1e-10, threading=true)
        new(model, α, A, M, grad_A, grad_p, X, Y, λ_1, λ_2, ϵ, threading)
    end
end


function (a::_grad_cost_alpha)(_, A)
    @inline _help(x) = begin
        v = Φ(a.model, x)
        ret = dot(v, A, v)^(-a.α) * (v * v')
        return ret
    end
    # putt grad_A to zero
    a.grad_A .= 0.0
    # foldl(add_grad!, zip(a.X, a.Y) |> Transducers.Map(x->_help(x[1]) * x[2]^(a.α)), init=a.grad_A)
    if a.threading
        a.grad_A .= foldxt(add_grad!, zip(a.X, a.Y) |> Transducers.Map(x->_help(x[1]) * x[2]^(a.α)))
    else
        a.grad_A .= foldl(add_grad!, zip(a.X, a.Y) |> Transducers.Map(x->_help(x[1]) * x[2]^(a.α)))
    end
    a.grad_A .*= (1/length(a.Y)) * (-1/a.α)
    a.grad_A .= a.grad_A + (1/ a.α) * a.M
    _λ1_regularization_gradient!(a.grad_A, A, a.λ_1)
    _λ2_regularization_gradient!(a.grad_A, A, a.λ_2)
    return nothing
end


struct _grad_ML <: _grad_struct
    model
    A
    M           # integration matrix
    grad_A
    grad_p
    X
    λ_1
    λ_2
    ϵ
end

function (a::_grad_ML)(_, A)
    @inline _help(x) = begin
        v = Φ(a.model, x)
        ret = (1/dot(v, A, v)) * (v * v')
        return ret
    end
    # put grad_A to zero
    a.grad_A .= 0.0
    # for x in a.X
    #     a.grad_A .+= _help(x)
    # end
    # foldl(add_grad!, zip(a.X, a.Y) |> Transducers.Map(_help), init=a.grad_A)
    a.grad_A .= foldxt(add_grad!, a.X |> Transducers.Map(_help))
    a.grad_A .*= -(1/length(a.X))
    a.grad_A .= a.grad_A + a.M
    _λ1_regularization_gradient!(a.grad_A, A, a.λ_1)
    _λ2_regularization_gradient!(a.grad_A, A, a.λ_2)
    return nothing
end

struct _grad_KL <: _grad_struct
    model
    A
    M_int               # integration matrix
    grad_A
    grad_p
    X
    Y
    λ_1
    λ_2
    ϵ
end

function (a::_grad_KL)(_, A)
    @inline _help(x) = begin
        v = Φ(a.model, x)
        ret = (1/dot(v, A, v)) * (v * v')
        return ret
    end
    # put grad_A to zero
    a.grad_A .= 0.0
    # for (x, y) in zip(a.X, a.Y)
    #     a.grad_A .+= _help(x) * y
    # end
    # foldl(add_grad!, zip(a.X, a.Y) |> Transducers.Map(x->_help(x[1]) * x[2]), init=a.grad_A)
    a.grad_A .= foldxt(add_grad!, zip(a.X, a.Y) |> Transducers.Map(x->_help(x[1]) * x[2]))
    a.grad_A .*= -(1/length(a.X))
    a.grad_A .= a.grad_A + a.M_int
    _λ1_regularization_gradient!(a.grad_A, A, a.λ_1)
    _λ2_regularization_gradient!(a.grad_A, A, a.λ_2)
    return nothing
end

struct _grad_reversed_KL <: _grad_struct
    model
    grad_A
    X
    Y
    λ_1
    λ_2
end

function (a::_grad_reversed_KL)(_, A)
    @inline _help(x) = begin
        v = Φ(a.model, x)
        ret = (1/dot(v, A, v)) * (v * v')
        return ret
    end
    # put grad_A to zero
    a.grad_A .= 0.0
    # for (x, y) in zip(a.X, a.Y)
    #     a.grad_A .+= _help(x) * y
    # end
    # foldl(add_grad!, zip(a.X, a.Y) |> Transducers.Map(x->_help(x[1]) * x[2]), init=a.grad_A)
    a.grad_A .= foldxt(add_grad!, zip(a.X, a.Y) |> Transducers.Map(x->_help(x[1]) * x[2]))
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
    use_putinar::Bool
    D
    C
    function ManoptOptPropblem(M::Manifolds.AbstractManifold, cost_func, grad_cost!::_grad_struct)
        ManoptOptPropblem(M, cost_func, grad_cost!, :gradient_descent)
    end
    function ManoptOptPropblem(M::Manifolds.AbstractManifold, cost_func, grad_cost!::_grad_struct, algorithm::Symbol)
        ManoptOptPropblem(M, cost_func, grad_cost!, nothing, algorithm)
    end
    function ManoptOptPropblem(M::Manifolds.AbstractManifold, cost_func, grad_cost!::_grad_struct, grad_cost_M!, algorithm::Symbol)
        ManoptOptPropblem(M, cost_func, grad_cost!, grad_cost_M!, algorithm, false, nothing, nothing)
    end
    function ManoptOptPropblem(M::Manifolds.AbstractManifold, cost_func, grad_cost!::_grad_struct, grad_cost_M!, algorithm::Symbol, use_putinar::Bool, D, C)
        new(M, cost_func, grad_cost!, grad_cost_M!, algorithm, use_putinar, D, C)
    end
end

function ManoptOptPropblem(cost_func, grad_cost!::_grad_struct, N::Int;
                    algorithm=:gradient_descent,
                    use_putinar=false,
                    D=nothing, C=nothing)
    if use_putinar
        @assert D !== nothing "D must be provided for putinar"
        @assert C !== nothing "C must be provided for putinar"
        M = Manifolds.SymmetricPositiveDefinite(N)
        N_SoS_comb = [size(m[1], 2) for m in D]
        M_comb = [Manifolds.SymmetricPositiveDefinite(N_SoS) for N_SoS in N_SoS_comb]
        M = foldl(×, M_comb, init=M)
        return ManoptOptPropblem(
            M, cost_func, grad_cost!, nothing, algorithm, use_putinar, D, C
        )
    end
    M = Manifolds.SymmetricPositiveDefinite(N)
    ManoptOptPropblem(M, cost_func, grad_cost!, algorithm)
end

function _grad_cost_M!(prob::ManoptOptPropblem, M, grad_A, A)
    prob.grad_cost!(grad_A, A)
    ManifoldDiff.riemannian_gradient!(M, grad_A, A, prob.grad_cost!.grad_A)
    return nothing
end

function _p_to_A!(A, M::Manifolds.ProductManifold, p, D_list, coef_list)
    A .= p[M, 1]
    for (i, (D, C)) in enumerate(zip(D_list, coef_list))
        for (_D, c) in zip(D, C)
            A .+= c * _D * p[M, i+1] * _D'
        end
    end
    return nothing
end

function _p_to_A!(A, M::Manifolds.AbstractManifold, p, D_list, coef_list)
    A .= p
    return nothing
end

function _p_to_A(M::Manifolds.ProductManifold, p, D_list, coef_list)
    A = zeros(size(p[M, 1])...)
    _p_to_A!(A, M, p, D_list, coef_list)
    return A
end


function _p_to_A(M::Manifolds.AbstractManifold, p, D_list, coef_list)
    return p
end

function _grad_A_to_grad_p!(grad_p, M, grad_A, D_list, coef_list)
    grad_p[M, 1] = grad_A
    for (i, (D, C)) in enumerate(Iterators.zip(D_list, coef_list))
        mat = zeros(size(grad_p[M, i+1])...)
        for (_D, c) in Iterators.zip(D, C)
            mat .+= c * _D' * grad_A * _D
        end
        grad_p[M, i+1] = mat
    end
    return nothing
end

function _grad_p_cost_M!(prob::ManoptOptPropblem, M, grad_p, p, D_list, coef_list)
    _grad_p_cost_M!(prob.grad_cost!, M, grad_p, p, D_list, coef_list)
    return nothing
end


"""
Add penalty to matrices to avoid numerical issues of small eigenvalues.
"""
function _M_penalty(M::Manifolds.ProductManifold, A)
    return -mapreduce(i->logdet(A[M, i]), +, 1:length(M.manifolds))
end

function _M_penalty(M, A)
    return -logdet(A)
end

function _grad_M_penalty!(grad_cost!::_grad_struct, M::Manifolds.ProductManifold, p, len)
    for i=1:len
        grad_cost!.grad_p[M, i] -= grad_cost!.ϵ * inv(p[M, i])
    end
    return nothing
end

function _grad_M_penalty!(grad_cost!::_grad_struct, M::Manifolds.AbstractManifold, p, len)
    grad_cost!.grad_p .-= grad_cost!.ϵ * inv(p)
    return nothing
end

function _grad_p_cost_M!(grad_cost!::_grad_struct, M::Manifolds.ProductManifold, grad_p, p, D_list, coef_list)
    _p_to_A!(grad_cost!.A, M, p, D_list, coef_list)
    grad_cost!(grad_cost!.grad_A, grad_cost!.A)
    _grad_A_to_grad_p!(grad_cost!.grad_p, M, grad_cost!.grad_A, D_list, coef_list)
    _grad_M_penalty!(grad_cost!, M, p, length(coef_list)+1)
    ManifoldDiff.riemannian_gradient!(M, grad_p, p, grad_cost!.grad_p)
    return nothing
end


function _grad_p_cost_M!(grad_cost!::_grad_struct, M::Manifolds.AbstractManifold, grad_p, p, D_list, coef_list)
    # _p_to_A!(grad_cost!.A, M, p, D_list, coef_list)
    grad_cost!.A .= p
    grad_cost!(grad_cost!.grad_A, grad_cost!.A)
    # _grad_A_to_grad_p!(grad_cost!.grad_p, M, grad_cost!.grad_A, D_list, coef_list)
    grad_cost!.grad_p .= grad_cost!.grad_A
    _grad_M_penalty!(grad_cost!, M, p, length(coef_list)+1)
    ManifoldDiff.riemannian_gradient!(M, grad_p, p, grad_cost!.grad_p)
    return nothing
end

function optimize(prob::ManoptOptPropblem, A_init;
            maxit=1000, trace=false,
            mingrad_stop=1e-8,
            custom_stopping_criterion=nothing,
            stepsize=nothing)
    M = prob.M
    cost_func = if prob.use_putinar
        let prob=prob
            (M, A) -> prob.cost_func(M, _p_to_A(M, A, prob.D, prob.C))
        end
    else
        prob.cost_func
    end
    grad_cost_M! = if prob.grad_cost_M! === nothing && prob.use_putinar == false
        let prob=prob
            (M, grad_A, A) -> _grad_cost_M!(prob, M, grad_A, A)
        end
    elseif prob.grad_cost_M! === nothing && prob.use_putinar == true
        let prob=prob, D=prob.D, C=prob.C
            (M, grad_A, A) -> _grad_p_cost_M!(prob, M, grad_A, A, D, C)
        end
    else
        prob.grad_cost_M!
    end

    debug = if trace
            [:Iteration,(:Change, "|Δp|: %1.9f |"), 
            (:Cost, " F(x): %1.11f | "), 
            (:Stepsize, " s: %f | "), 1, "\n", :Stop]
    else
        []
    end
    
    stopping_criterion = if custom_stopping_criterion === nothing
        (Manopt.StopAfterIteration(maxit) | Manopt.StopWhenGradientNormLess(mingrad_stop) | Manopt.StopWhenStepsizeLess(1e-8))
    else
        custom_stopping_criterion
    end

    _stepsize = if stepsize === nothing
        if prob.algorithm == :quasi_newton
            Manopt.default_stepsize(M, Manopt.QuasiNewtonState)
        elseif prob.algorithm == :gradient_descent
            Manopt.default_stepsize(M, Manopt.GradientDescentState)
        else
            throw(error("No optimal step size for $(prob.algorithm) known, set one explicitly."))
        end
    else
        stepsize
    end

    A_sol = deepcopy(A_init)
    if A_sol isa Hermitian || A_sol isa Symmetric
        A_sol = Matrix(A_sol)
    end
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
    elseif prob.algorithm == :quasi_newton
        Manopt.quasi_Newton!(M, cost_func, grad_cost_M!, A_sol,
                evaluation=Manopt.InplaceEvaluation(),
                stopping_criterion=stopping_criterion,
                # cautious_update=true,
                # direction_update=Manopt.InverseSR1(),
                debug=debug,
                stepsize=_stepsize,
                # memory_size=5
                )
    else
        throw(error("Algorithm $(prob.algorithm) not implemented."))
        return nothing
    end
    return A_sol
end



function _α_divergence_Manopt!(a::PSDModel{T},
                α::T,
                X::PSDDataVector{T},
                Y::AbstractVector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
                ϵ = 1e-8,
                use_putinar=true,
                trace=false,
                normalization=false,
                algorithm=:quasi_newton,
                threading=true,
                use_CV=false,
                CV_K=5,
                maxit=1000,
                mingrad_stop=1e-8,
                kwargs...
            ) where {T<:Number}

    @assert normalization == false "Normalization not implemented yet."
    @assert α != 1 "Use KL divergence instead"
    @assert α != 0 "Use reversed KL divergence instead"

    d = length(X[1])

    D_list, coef_list = if typeof(a) <: PSDModelPolynomial
        get_semialgebraic_domain_constraints(a)
    else
        use_putinar = false
        [], []
    end

    N = size(a.B, 1)

    M = if use_putinar
        M_list = []
        for i in 1:d
            push!(M_list, Manifolds.SymmetricPositiveDefinite(size(D_list[i][1],2)))
        end
        foldl(×, M_list, init=Manifolds.SymmetricPositiveDefinite(N))
    else
        Manifolds.SymmetricPositiveDefinite(N)
    end


    function _cost_alpha(A, M_int, α, X, Y)
        res = zero(T)
        for (x, y) in zip(X, Y)
            v = Φ(a, x)
            res += dot(v, A, v)^(1-α) * y^(α)
        end
        y_int = (1/length(Y)) * (1/(α-1)) * sum(Y)
        res = (1/length(Y)) * (1/(α * (α - 1))) * res + (1/α)* tr(A * M_int) - y_int
        res += λ_1 * _λ1_regularization(A) + λ_2 * _λ2_regularization(A)
    end

    stop_CV = nothing
    X_train, Y_train = X, Y
    X_test, Y_test = nothing, nothing
    
    if use_CV
        shuf = Random.shuffle(1:length(X))
        _Xtmp, _Ytmp = X[shuf], Y[shuf]
        _X = [_Xtmp[1+round(Int, ((k-1)/CV_K) * length(X)):round(Int, (k/CV_K) * length(X))] for k in 1:CV_K]
        _Y = [_Ytmp[1+round(Int, ((k-1)/CV_K) * length(X)):round(Int, (k/CV_K) * length(X))] for k in 1:CV_K]
    end
    
    p = rand(M)
    M_int = integration_matrix(a)
    for k=1:CV_K

        if use_CV
            trace && println("CV iteration $k of $CV_K")
            X_train = [_X[k2] for k2 in 1:CV_K if k2 != k] |> x->vcat(x...)
            X_test = _X[k]
            Y_train = [_Y[k2] for k2 in 1:CV_K if k2 != k] |> x->vcat(x...)
            Y_test = _Y[k]
            CV_loss = let α=α, M_int=M_int
                (X, Y, A) -> _cost_alpha(A, M_int, α, X, Y)
            end
            stop_CV = CrossValidationStopping(X_test, Y_test, CV_loss; coef_list=coef_list, D_list=D_list) | 
                        Manopt.StopAfterIteration(maxit) |
                        Manopt.StopWhenGradientNormLess(mingrad_stop) |
                        Manopt.StopWhenStepsizeLess(1e-8)
        end

        cost_alpha = let α=α, M_int=M_int, X_train=X_train, Y_train=Y_train, D_list=D_list, coef_list=coef_list
            (M, A) -> _cost_alpha(_p_to_A(M, A, D_list, coef_list), M_int, α, X_train, Y_train) + ϵ * _M_penalty(M, A)
        end

        grad_p = rand(M)
        grad_alpha! = _grad_cost_alpha(a, α, zeros(T, N, N), M_int, zeros(T, N, N), grad_p, X_train, Y_train, λ_1, λ_2; ϵ=ϵ, threading=threading)
        grad_p_M! = let grad_t=grad_alpha!, D_list=D_list, coef_list=coef_list
            (M, grad_p, p) -> _grad_p_cost_M!(grad_t, M, grad_p, p, D_list, coef_list)
        end
        

        prob = ManoptOptPropblem(M, cost_alpha, grad_alpha!, grad_p_M!, algorithm)
        p = optimize(prob, p; trace=trace, custom_stopping_criterion=stop_CV, kwargs...)
        
    end
    A_new = _p_to_A(M, p, D_list, coef_list)
    set_coefficients!(a, Hermitian(A_new))
    return _cost_alpha(A_new, M_int, α, X, Y)
end


function _adaptive_α_divergence_Manopt!(a::PSDModelOrthonormal{d, T},
                α::T,
                X::PSDDataVector{T},
                Y::AbstractVector{T},
                g::Function, adap_s::SamplingStruct{T, d};
                λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                normalization=false,
                algorithm=:gradient_descent,
                maxit=1000,
                adaptive_sample_steps=10,
                broadcasted_target = false,
                kwargs...
            ) where {d, T<:Number}

    @assert normalization == false "Normalization not implemented yet."
    @assert α != 1 "Use KL divergence instead"
    @assert α != 0 "Use reversed KL divergence instead"

    @assert δ > 0 "δ must be positive"
    @assert 0 < p < 1 "p must be in (0, 1)"

    function _cost_alpha(A, M_int, α, X, Y)
        res = zero(T)
        for (x, y) in zip(X, Y)
            v = Φ(a, x)
            res += dot(v, A, v)^(1-α) * y^(α)
        end
        y_int = (1/length(Y)) * (1/(α-1)) * sum(Y)
        res = (1/length(Y)) * (1/(α * (α - 1))) * res + (1/α)* tr(A * M_int) - y_int
        res += λ_1 * _λ1_regularization(A) + λ_2 * _λ2_regularization(A)
    end

    M_int = integration_matrix(a)
    N_sample = length(Y)
    for i in 1:adaptive_sample_steps
        loss = (x, y) -> begin
            # res = zero(T)
            v = Φ(a, x)
            res = dot(v, a.B, v)^(1-α) * y^(α)
            y_int = (1/(α-1)) * y
            res = (1/(α * (α - 1))) * res + (1/α)* res - (1/(α-1)) * y_int
            # res += λ_1 * _λ1_regularization(A) + λ_2 * _λ2_regularization(A)
            return res
        end
        # do not sample adaptive in first round, variance too unreliable
        
        sample!(adap_s, g, loss, X, Y; broadcasted_target=broadcasted_target)

        trace && println("Iteration $i: $(length(Y)) samples, added $(length(Y) - N_sample) samples.")
        N_sample = length(Y)

        cost_alpha = let α=α, X=X, Y=Y, M_int=M_int
            (M, A) -> _cost_alpha(A, M_int, α, X, Y)
        end

        N = size(a.B, 1)

        grad_alpha! = _grad_cost_alpha(a, α, zeros(T, N, N), M_int, zeros(T, N, N), zeros(T, N, N), X, Y, λ_1, λ_2)

        prob = ManoptOptPropblem(cost_alpha, grad_alpha!, N, algorithm=algorithm)
        A_new = optimize(prob, a.B; trace=trace, maxit=Int(floor(maxit/adaptive_sample_steps)), kwargs...)
        set_coefficients!(a, A_new)
    end
    return _cost_alpha(a.B, M_int, α, X, Y)
end


function _adaptive_CV_α_divergence_Manopt!(a::PSDModelOrthonormal{d, T},
                α::T,
                X::PSDDataVector{T},
                Y::AbstractVector{T},
                g::Function,
                samp_t::SamplingStruct{T, d};
                λ_1 = 0.0,
                λ_2 = 0.0,
                ϵ = 1e-7,
                trace=false,
                normalization=false,
                algorithm=:gradient_descent,
                adaptive_sample_steps=10,
                CV_split=0.8,
                broadcasted_target = false,
                maxit=1000,
                mingrad_stop=1e-8,
                threading=true,
                normalize_data=true,
                kwargs...
            ) where {d, T<:Number}

    @assert normalization == false "Normalization not implemented yet."
    @assert α != 1 "Use KL divergence instead"
    @assert α != 0 "Use reversed KL divergence instead"

    M_int = integration_matrix(a)

    function _cost_alpha(A, M_int, α, X, Y)
        res = zero(T)
        for (x, y) in zip(X, Y)
            v = Φ(a, x)
            res += dot(v, A, v)^(1-α) * y^(α)
        end
        y_int = (1/length(Y)) * (1/(α-1)) * sum(Y)
        res = (1/length(Y)) * (1/(α * (α - 1))) * res + (1/α)* tr(A) - y_int
        res += λ_1 * _λ1_regularization(A) + λ_2 * _λ2_regularization(A)
    end
    
    CV_loss(X, Y, A) = _cost_alpha(A, M_int, α, X, Y)

    for _ in 1:adaptive_sample_steps
        sample!(samp_t, g, (x, y)->CV_loss([x], [y], a.B), X, Y; broadcasted_target=broadcasted_target, trace=trace)
        # split into train and test, with 10 percent test data
        _Y_copy = if normalize_data
            Y .* (length(Y) / sum(Y))
        else
            Y
        end
        shuf = Random.shuffle(1:length(X))
        _X, _Y = X[shuf], _Y_copy[shuf]
        X_train = _X[1:round(Int, CV_split * length(_X))]
        Y_train = _Y[1:round(Int, CV_split * length(_Y))]
        X_test = _X[round(Int, CV_split * length(_X))+1:end]
        Y_test = _Y[round(Int, CV_split * length(_Y))+1:end]

        cost_alpha = let α=α, X=X_train, Y=Y_train
            (M, A) -> _cost_alpha(A, M_int, α, X, Y) + ϵ * _M_penalty(M, A)
        end

        N = size(a.B, 1)

        stop_CV = CrossValidationStopping(X_test, Y_test, CV_loss) | 
                    Manopt.StopAfterIteration(maxit) |
                    Manopt.StopWhenGradientNormLess(mingrad_stop) |
                    Manopt.StopWhenStepsizeLess(1e-8)

        grad_alpha! = _grad_cost_alpha(a, α, zeros(T, N, N), M_int, zeros(T, N, N), zeros(T, N, N),
                        X_train, Y_train, λ_1, λ_2; threading=threading)

        prob = ManoptOptPropblem(cost_alpha, grad_alpha!, N, algorithm=algorithm)
        A_new = optimize(prob, a.B; trace=trace, 
                custom_stopping_criterion=stop_CV, kwargs...)
        set_coefficients!(a, Hermitian(A_new))
    end
    return _cost_alpha(a.B, M_int, α, X, Y)
end


function _ML_Manopt!(a::PSDModel{T},
                X::PSDDataVector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
                ϵ = 1e-7,
                trace=false,
                normalization=false,
                algorithm=:gradient_descent,
                use_putinar=true,
                maxit=1000,
                mingrad_stop=1e-8,
                CV_K=5,
                use_CV=false,
                kwargs...
            ) where {T<:Number}

    @assert normalization == false "Normalization not implemented yet."
    d = length(X[1])
    function _cost_ML(A, M_int, X)
        res = zero(T)
        for x in X
            # v = Φ(a, x)
            res += -log(a(x, A))
        end
        res = (1/length(X)) * res + tr(A * M_int)
        res += λ_1 * _λ1_regularization(A) + λ_2 * _λ2_regularization(A)
        return res
    end

    D_list, coef_list = if typeof(a) <: PSDModelPolynomial
        get_semialgebraic_domain_constraints(a)
    else
        use_putinar = false
        [], []
    end

    stop_CV = nothing
    X_train = X
    X_test = nothing
    _X = nothing

    if use_CV
        shuf = Random.shuffle(1:length(X))
        _Xtmp = X[shuf]
        _X = [_Xtmp[1+round(Int, ((k-1)/CV_K) * length(X)):round(Int, (k/CV_K) * length(X))] for k in 1:CV_K]
    end

    N = size(a.B, 1)

    M = if use_putinar
        M_list = []
        for i in 1:d
            push!(M_list, Manifolds.SymmetricPositiveDefinite(size(D_list[i][1],2)))
        end
        foldl(×, M_list, init=Manifolds.SymmetricPositiveDefinite(N))
    else
        Manifolds.SymmetricPositiveDefinite(N)
    end

    p = rand(M)
    M_int = integration_matrix(a)

    for k =1:CV_K
        if use_CV
            trace && println("CV iteration $k of $CV_K")
            X_train = [_X[k2] for k2 in 1:CV_K if k2 != k] |> x->vcat(x...)
            X_test = _X[k]
            CV_loss = (X, Y, A) -> _cost_ML(A, M_int, X)
            stop_CV = CrossValidationStopping(X_test, T[], CV_loss; coef_list=coef_list, D_list=D_list) | 
                        Manopt.StopAfterIteration(maxit) |
                        Manopt.StopWhenGradientNormLess(mingrad_stop) |
                        Manopt.StopWhenStepsizeLess(1e-8)
        end

        cost_ML = let X_train=X_train, M_int=M_int
                (M, A) -> _cost_ML(_p_to_A(M, A, D_list, coef_list), M_int, X_train) + ϵ * _M_penalty(M, A)
            end

        N = size(a.B, 1)

        grad_ML! = _grad_ML(a, zeros(T, N, N), M_int, zeros(T, N, N), rand(M), X_train, λ_1, λ_2, ϵ)
        grad_p_M! = let grad_t=grad_ML!, D_list=D_list, coef_list=coef_list
            (M, grad_p, p) -> _grad_p_cost_M!(grad_t, M, grad_p, p, D_list, coef_list)
        end

        prob = ManoptOptPropblem(M, cost_ML, grad_ML!, grad_p_M!, algorithm)
        
        p = optimize(prob, p; trace=trace, maxit=maxit, custom_stopping_criterion=stop_CV, kwargs...)
        if !use_CV
            break
        end
    end
    A_new = _p_to_A(M, p, D_list, coef_list)
    set_coefficients!(a, Hermitian(A_new))
    return _cost_ML(A_new, M_int, X)
end


function _KL_Manopt!(a::PSDModel{T},
                X::PSDDataVector{T},
                Y::AbstractVector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
                ϵ = 1e-8,
                trace=false,
                normalization=false,
                algorithm=:gradient_descent,
                use_putinar=true,
                use_CV=false,
                mingrad_stop=1e-8,
                maxit=1000,
                CV_K=5,
                kwargs...
            ) where {T<:Number}

    @assert normalization == false "Normalization not implemented yet."

    d = length(X[1])
    N = size(a.B, 1)

    D_list, coef_list = if typeof(a) <: PSDModelPolynomial
        get_semialgebraic_domain_constraints(a)
    else
        use_putinar = false
        [], []
    end

    M = if use_putinar
        M_list = []
        for i in 1:d
            push!(M_list, Manifolds.SymmetricPositiveDefinite(size(D_list[i][1],2)))
        end
        foldl(×, M_list, init=Manifolds.SymmetricPositiveDefinite(N))
    else
        Manifolds.SymmetricPositiveDefinite(N)
    end

    function _cost_KL(A, M_int, X, Y)
        res = zero(T)
        for (x, y) in zip(X, Y)
            res += (log(y)-log(a(x, A))-1) * y
        end
        res = (1/length(X)) * res + tr(A * M_int)
        res += λ_1 * _λ1_regularization(A) + λ_2 * _λ2_regularization(A)
        return res
    end

    stop_CV = nothing
    X_train, Y_train = X, Y
    X_test, Y_test = nothing, nothing
    
    if use_CV
        shuf = Random.shuffle(1:length(X))
        _Xtmp, _Ytmp = X[shuf], Y[shuf]
        _X = [_Xtmp[1+round(Int, ((k-1)/CV_K) * length(X)):round(Int, (k/CV_K) * length(X))] for k in 1:CV_K]
        _Y = [_Ytmp[1+round(Int, ((k-1)/CV_K) * length(X)):round(Int, (k/CV_K) * length(X))] for k in 1:CV_K]
    end
    
    p = rand(M)
    M_int = integration_matrix(a)
    for k=1:CV_K

        if use_CV
            trace && println("CV iteration $k of $CV_K")
            X_train = [_X[k2] for k2 in 1:CV_K if k2 != k] |> x->vcat(x...)
            X_test = _X[k]
            Y_train = [_Y[k2] for k2 in 1:CV_K if k2 != k] |> x->vcat(x...)
            Y_test = _Y[k]
            CV_loss = let M_int=M_int
                (X, Y, A) -> _cost_KL(A, M_int, X, Y)
            end
            stop_CV = CrossValidationStopping(X_test, Y_test, CV_loss; coef_list=coef_list, D_list=D_list) | 
                        Manopt.StopAfterIteration(maxit) |
                        Manopt.StopWhenGradientNormLess(mingrad_stop) |
                        Manopt.StopWhenStepsizeLess(1e-8)
        end

        cost_KL = let M_int=M_int, X_train=X_train, Y_train=Y_train, D_list=D_list, coef_list=coef_list
            (M, A) -> _cost_KL(_p_to_A(M, A, D_list, coef_list), M_int, X_train, Y_train) + ϵ * _M_penalty(M, A)
        end

        grad_p = rand(M)
        grad_KL! = _grad_KL(a, zeros(T, N, N), M_int, zeros(T, N, N), grad_p, X_train, Y_train, λ_1, λ_2, ϵ)
        grad_p_M! = let grad_t=grad_KL!, D_list=D_list, coef_list=coef_list
            (M, grad_p, p) -> _grad_p_cost_M!(grad_t, M, grad_p, p, D_list, coef_list)
        end
        

        prob = ManoptOptPropblem(M, cost_KL, grad_KL!, grad_p_M!, algorithm)
        p = optimize(prob, p; trace=trace, custom_stopping_criterion=stop_CV, kwargs...)
        
    end
    A_new = _p_to_A(M, p, D_list, coef_list)
    set_coefficients!(a, Hermitian(A_new))
    return _cost_KL(A_new, M_int, X, Y)
end