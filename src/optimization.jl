using ProximalOperators: IndPSD, prox, prox!
import ProximalAlgorithms

const _optimize_PSD_kwargs = 
    (:convex, :trace, :maxit, :tol, 
    :smooth, :opt_algo, :vectorize_matrix, 
    :normalization_constraint, :optimizer)

"""
optimize_PSD_model(initial::AbstractMatrix, 
                    loss::Function, 
                    X::AbstractVector;
                    λ_1::Float64=1e-8,
                    trace::Bool=false,
                    maxit=5000,
                    tol=1e-6,
                    smooth=true,
                ) where {T<:Number}

Minimizes loss with the constraint of PSD and chooses the right
solver depending on the model.
"""
function optimize_PSD_model(initial::AbstractMatrix{T}, 
                    loss::Function;
                    convex::Bool = true,
                    trace::Bool=false,
                    maxit::Int=5000,
                    tol::Real=1e-6,
                    smooth::Bool=true,
                    opt_algo=nothing,
                    optimizer=nothing,
                    vectorize_matrix::Bool=true,
                    normalization_constraint::Bool=false,
                    fixed_variables=nothing,
                ) where {T<:Number}
    if convex
        return optimize_PSD_model_convex(initial, loss;
                trace=trace,
                maxit=maxit,
                optimizer=optimizer,
                normalization_constraint=normalization_constraint,
                fixed_variables=fixed_variables,
            )
    end

    if normalization_constraint
        throw(error("Only implemented for the convex case."))
    end

    if fixed_variables !== nothing
        throw(error("Only implemented for the convex case."))
    end
    # set default parameters
    # TODO

    verbose_solver = trace ? true : false
    solver = if opt_algo !== nothing
        opt_algo(maxit=maxit, tol=tol, verbose=verbose_solver)
    elseif !convex && smooth
        ProximalAlgorithms.ForwardBackward(maxit=maxit, tol=tol, verbose=verbose_solver)
    else
        @error "Not convex and not smooth is not implemented yet."
    end

    N = size(initial, 1)
    if vectorize_matrix
        psd_constraint = IndPSD(scaling=true)
        view_mat = view_mat_for_to_symmetric(N)
        loss2(x::AbstractVector) = loss(low_vec_to_Symmetric(x, view_mat))
        solution, _ = solver(x0=Hermitian_to_low_vec(initial), f=loss2, g=psd_constraint)
        return Hermitian(copy(low_vec_to_Symmetric(solution, view_mat)), :L)
    else
        psd_constraint = IndPSD(scaling=false)
        solution, _ = solver(x0=Matrix(initial), f=loss, g=psd_constraint)
        return Hermitian(solution)
    end
end

import Convex as con
using Convex: opnorm
import SCS
## utils Convex for least squares
Base.:^(x::con.AbstractExpr, p::Int) = begin
    @assert p == 2
    return con.sumsquares(x)
end

Base.:/(x::Number, y::con.AbstractExpr) = begin
    return x * con.invpos(y)
end

# nuclearnorm(A::AbstractMatrix) = sum(svdvals(A))
nuclearnorm(A::con.AbstractExpr) = con.nuclearnorm(A)

function optimize_PSD_model_convex(initial::AbstractMatrix{T}, 
                    loss::Function;
                    trace::Bool=false,
                    maxit::Int=5000,
                    normalization_constraint=false,
                    optimizer=nothing,
                    fixed_variables=nothing,
                ) where {T<:Number}
    verbose_solver = trace ? true : false

    if optimizer === nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            SCS.Optimizer, 
            "max_iters" => maxit
        )
    else
        @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
    end

    N = size(initial, 1)
    B = con.Variable((N, N))
    B.value = initial
    if fixed_variables !== nothing
        con.fix!(B[fixed_variables...], initial[fixed_variables...])
    end
    problem = con.minimize(loss(B), con.isposdef(B))
    if normalization_constraint
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        problem.constraints += con.tr(B) == 1
    end

    con.solve!(problem,
        optimizer;
        silent_solver = !verbose_solver
    )
    return Hermitian(T.(con.evaluate(B)))
end

function iteratively_optimize_convex(initial::AbstractMatrix{T}, 
                    loss::Function,
                    convergence::Function;
                    trace::Bool=false,
                    maxit::Int=5000,
                    convergence_tol::Real=1e-6,
                    normalization_constraint=false,
                    optimizer=nothing,
                ) where {T<:Number}
    verbose_solver = trace ? true : false

    if optimizer === nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            SCS.Optimizer, 
            "max_iters" => maxit
        )
    else
        @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
    end

    N = size(initial, 1)
    B = con.Variable((N, N))
    con.set_value!(B, initial)

    # B2 for fixed B_{k-1}
    B2  = con.Variable((N, N))
    # con.set_value!(B2, initial)
    con.fix!(B2, initial)
    problem = con.minimize(loss(B, B2), con.isposdef(B))
    if normalization_constraint
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        problem.constraints += con.tr(B) == 1
    end

    # con.fix!(B2)
    # 1st solve
    con.solve!(problem,
        optimizer;
        silent_solver = !verbose_solver
    )
    k = 1
    while !convergence(con.evaluate(B), con.evaluate(B2), k)
        # con.set_value!(B2, con.evaluate(B))
        con.fix!(B2, con.evaluate(B))
        con.solve!(problem,
            optimizer;
            silent_solver = !verbose_solver,
            warmstart = true
        )
        k = k + 1
    end
    return Hermitian(T.(con.evaluate(B)))
end