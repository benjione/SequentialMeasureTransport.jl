import Manopt
import Manifolds

struct ManOptProp{T} <: OptProp{T}
    initial::AbstractMatrix{T}
    loss::Function
    grad_loss!::Function
    trace::Bool
    function ManOptProp(
            initial::AbstractMatrix{T}, 
            loss::Function, 
            grad_loss!::Function;
            trace=false
        ) where {T<:Number}
        new{T}(initial, loss, grad_loss!, trace)
    end
end
function ManOptProp(
            initial::AbstractMatrix{T}, 
            loss::Function;
            kwargs...
        ) where {T<:Number}
    grad_loss! = let loss = loss 
        (M, G, x) -> begin
            G .= Manifolds.project(M, x, FD.gradient(loss, x))
            return G
        end
    end
    loss_manopt = let loss=loss
        (M, x) -> loss(x)
    end
    ManOptProp(initial, loss_manopt, grad_loss!; kwargs...)
end

function optimize(
        prob::ManOptProp{T}
    ) where {T<:Number}
    n = size(prob.initial, 1)
    M = Manifolds.SymmetricPositiveDefinite(n)
    x0 = Matrix(prob.initial)
    # x0 = Manifolds.rand(M)
    Manopt.gradient_descent!(M, prob.loss,
            prob.grad_loss!, x0;
            evaluation=Manopt.InplaceEvaluation(),
            stepsize=Manopt.ArmijoLinesearch(
                M;
                initial_stepsize=1.0,
                contraction_factor=0.9,
                sufficient_decrease=0.05,
                stop_when_stepsize_less=1e-9,
            ),
            debug=[:Iteration,(:Change, "|Δp|: %1.10f |"),(:GradientChange, "|Δg|: %1.10f |"),
                    (:Cost, " F(x): %1.11f | "), "\n", :Stop],
            stopping_criterion = Manopt.StopWhenGradientChangeLess(1e-10) | Manopt.StopAfterIteration(200),
        )
    return Hermitian(x0)
end