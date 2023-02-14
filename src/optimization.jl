

const _optimize_PSD_kwargs = 
    (:convex, :trace, :maxit, :tol, :smooth, :opt_algo, :vectorize_matrix)

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
                    convex = true,
                    trace::Bool=false,
                    maxit::Int=5000,
                    tol::Real=1e-6,
                    smooth::Bool=true,
                    opt_algo=nothing,
                    vectorize_matrix::Bool=true,
                    )

    verbose_solver = trace ? true : false
    solver = if opt_algo !== nothing
        opt_algo(maxit=maxit, tol=tol, verbose=verbose_solver)
    elseif convex && smooth
        ProximalAlgorithms.FastForwardBackward(maxit=maxit, tol=tol, verbose=verbose_solver)
    elseif convex && !smooth
        ProximalAlgorithms.PANOCplus(gamma=1e-5, maxit=maxit, tol=tol, verbose=verbose_solver)
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