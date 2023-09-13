import Convex as con
using Convex: opnorm
import SCS

struct SDPOptProp{T} <: OptProp{T}
    initial::AbstractMatrix{T}
    loss::Function
    normalization::Bool           # if tr(X) = 1
    optimizer
    fixed_variables
    trace::Bool
    function SDPOptProp(
            initial::AbstractMatrix{T}, 
            loss::Function;
            trace=false,
            optimizer=nothing,
            fixed_variables=nothing,
            normalization=false,
            maxit::Int=5000,
        ) where {T<:Number}
        if optimizer === nothing
            optimizer = con.MOI.OptimizerWithAttributes(
                SCS.Optimizer, 
                "max_iters" => maxit
            )
        else
            @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
        end
        new{T}(initial, 
                loss,
                normalization,
                optimizer,
                fixed_variables,
                trace
            )
    end
end

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

function optimize(prob::SDPOptProp{T}) where {T<:Number}
    verbose_solver = prob.trace ? true : false

    N = size(prob.initial, 1)
    B = con.Variable((N, N))
    B.value = prob.initial
    if prob.fixed_variables !== nothing
        con.fix!(B[prob.fixed_variables...], prob.initial[prob.fixed_variables...])
    end
    problem = con.minimize(prob.loss(B), con.isposdef(B))
    if prob.normalization
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        problem.constraints += con.tr(B) == 1
    end

    con.solve!(problem,
        prob.optimizer;
        silent_solver = !verbose_solver
    )
    return Hermitian(T.(con.evaluate(B)))
end


# function iteratively_optimize_convex(initial::AbstractMatrix{T}, 
#                     loss::Function,
#                     convergence::Function;
#                     trace::Bool=false,
#                     maxit::Int=5000,
#                     convergence_tol::Real=1e-6,
#                     normalization_constraint=false,
#                     optimizer=nothing,
#                 ) where {T<:Number}
#     verbose_solver = trace ? true : false

#     if optimizer === nothing
#         optimizer = con.MOI.OptimizerWithAttributes(
#             SCS.Optimizer, 
#             "max_iters" => maxit
#         )
#     else
#         @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
#     end

#     N = size(initial, 1)
#     B = con.Variable((N, N))
#     con.set_value!(B, initial)

#     # B2 for fixed B_{k-1}
#     B2  = con.Variable((N, N))
#     # con.set_value!(B2, initial)
#     con.fix!(B2, initial)
#     problem = con.minimize(loss(B, B2), con.isposdef(B))
#     if normalization_constraint
#         # IMPORTANT: only valid for tensorized polynomial maps.
#         @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
#         problem.constraints += con.tr(B) == 1
#     end

#     # con.fix!(B2)
#     # 1st solve
#     con.solve!(problem,
#         optimizer;
#         silent_solver = !verbose_solver
#     )
#     k = 1
#     while !convergence(con.evaluate(B), con.evaluate(B2), k)
#         # con.set_value!(B2, con.evaluate(B))
#         con.fix!(B2, con.evaluate(B))
#         con.solve!(problem,
#             optimizer;
#             silent_solver = !verbose_solver,
#             warmstart = true
#         )
#         k = k + 1
#     end
#     return Hermitian(T.(con.evaluate(B)))
# end