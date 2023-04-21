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
function optimize_PSD_model(initial::AbstractMatrix, 
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
                )
    if convex
        return optimize_PSD_model_convex(initial, loss;
                trace=trace,
                maxit=maxit,
                tol=tol,
                optimizer=optimizer,
                smooth=smooth,
                normalization_constraint=normalization_constraint,
            )
    end

    if normalization_constraint
        @error "Only implemented for the convex case."
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

nuclearnorm(A::AbstractMatrix) = sum(svdvals(A))
nuclearnorm(A::con.AbstractExpr) = con.nuclearnorm(A)

function optimize_PSD_model_convex(initial::AbstractMatrix, 
                    loss::Function;
                    trace::Bool=false,
                    maxit::Int=5000,
                    tol::Real=1e-6,
                    smooth::Bool=true,
                    normalization_constraint=false,
                    optimizer=nothing,
                )
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
    return Hermitian(con.evaluate(B))
end