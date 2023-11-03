abstract type OptProp{T} end

include("SDP_optimizer.jl")
include("manopt_optimizer.jl")

const _optimize_PSD_kwargs = 
    (:convex, :trace, :maxit, :tol, 
    :smooth, :opt_algo, :vectorize_matrix, 
    :normalization_constraint, :optimizer,
    :optimization_method)

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
function create_SoS_opt_problem(
            method::Symbol,
            initial::AbstractMatrix{T}, 
            loss::Function;
            kwargs...
        ) where {T<:Number}
    if method == :manopt
        return create_SoS_manopt_problem(initial, loss; kwargs...)
    elseif method == :SDP
        return create_SoS_SDP_problem(initial, loss; kwargs...)
    else
        throw(error("Optimization method $method not implemented."))
        return nothing
    end
end

function create_SoS_manopt_problem(
            initial::AbstractMatrix{T}, 
            loss::Function;
            grad_loss=nothing,
            convex::Bool = true,
            trace::Bool=false,
            maxit::Int=5000,
            tol::Real=1e-6,
            optimizer=nothing,
            vectorize_matrix::Bool=true,
            normalization_constraint::Bool=false,
            fixed_variables=nothing,
        ) where {T<:Number}
    if grad_loss === nothing
        return ManOptProp(initial, loss)
    else
        return ManOptProp(initial, loss, grad_loss)
    end
end

function create_SoS_SDP_problem(
            initial::AbstractMatrix{T}, 
            loss::Function;
            trace::Bool=false,
            maxit::Int=5000,
            optimizer=nothing,
            normalization_constraint::Bool=false,
            fixed_variables=nothing,
        ) where {T<:Number}
    return SDPOptProp(initial, loss; 
            trace=trace,
            maxit=maxit,
            optimizer=optimizer,
            normalization=normalization_constraint,
            fixed_variables=fixed_variables,
        )
end