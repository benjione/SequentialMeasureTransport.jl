module OptimalTransport

using ..SequentialMeasureTransport
import ..SequentialMeasureTransport as SMT
using ..SequentialMeasureTransport: PSDDataVector
using Distributions
using FastGaussQuadrature: gausslegendre


function entropic_OT!(model::SMT.PSDModelOrthonormal{d2, T},
        cost::Function,
        p::Function,
        q::Function,
        ϵ::T,
        XY::PSDDataVector{T};
        X=nothing, Y=nothing,
        preconditioner::Union{<:SMT.ConditionalMapping{d2, 0, T}, Nothing}=nothing,
        reference::Union{<:SMT.ReferenceMap{d2, 0, T}, Nothing}=nothing,
        use_putinar=true,
        use_preconditioner_cost=false,
        λ_marg=nothing,
        kwargs...) where {d2, T<:Number}
    @assert d2 % 2 == 0
    d = d2 ÷ 2
    reverse_KL_cost = begin
        if use_preconditioner_cost
            let p=p, q=q
                x->p(x[1:d]) * q(x[d+1:end])
            end
        else
            _rev_KL_density = let p=p, q=q
                x -> p(x[1:d]) * q(x[d+1:end])
            end
            if reference !== nothing
                _rev_KL_density = SMT.pushforward(reference, _rev_KL_density)
            end
            if preconditioner === nothing
                _rev_KL_density
            else
                SMT.pullback(preconditioner, _rev_KL_density)
            end
        end
    end

    cost_pb = begin
        _cost = let cost=cost
                x -> cost(x)
        end
        if reference !== nothing
            _cost = SMT.pushforward(reference, _cost)
        else
            _cost
        end
    end

    if preconditioner !== nothing
        cost_pb = let cost_pb=cost_pb
            x -> cost_pb(SMT.pushforward(preconditioner, x))
        end
    end

    ξ = map(x->reverse_KL_cost(x), XY)
    ξ2 = map(x->cost_pb(x), XY)
    if λ_marg === nothing
        ## estimate the order of the reverse KL cost to find an acceptable λ_marg
        ## to do that, we calculate KL(U||reverse_KL_cost) where U is the distribution of XY
        _order_rev_KL = (sum(ξ2) - ϵ * sum(log.(ξ))) / length(ξ)
        λ_marg = 10.0*_order_rev_KL
        @info "Estimated order of the reverse KL cost: $_order_rev_KL \n 
                Setting λ_marg to $λ_marg"
    end

    model_for_marg = if preconditioner === nothing
        model
    else
        SMT._add_mapping(model, preconditioner)
    end
    if X === nothing
        X = [x[1:d] for x in XY]
    end
    if Y === nothing
        Y = [x[d+1:end] for x in XY]
    end
    

    _p, _q = if reference !== nothing
        _p = SMT.pushforward(reference[1:d], p)
        _q = SMT.pushforward(reference[d+1:end], q)
        _p, _q
    else
        p, q
    end

    ## pushforward the samples
    if reference !== nothing
        _XY_marg = SMT.pushforward.(Ref(reference), [[x;y] for (x, y) in zip(X, Y)])
        X = [x[1:d] for x in _XY_marg]
        Y = [x[d+1:end] for x in _XY_marg]
    end
    
    ## evaluate the marginals on the original samples
    p_X = map(_p, X)
    q_Y = map(_q, Y)
    e_X = collect(1:d)
    e_Y = collect(d+1:d2)
    if use_putinar && (typeof(model) <: SMT.PSDModelPolynomial)
        D, C = SMT.get_semialgebraic_domain_constraints(model)
        return SMT._OT_JuMP!(model, cost_pb, ϵ, XY, ξ; mat_list=D, coef_list=C, 
                model_for_marginals=model_for_marg,
                marg_regularization = [(e_X, X, p_X), (e_Y, Y, q_Y)],
                λ_marg_reg=λ_marg,
                kwargs...)
    else
        return SMT._OT_JuMP!(model, cost_pb, ϵ, XY, ξ; 
                model_for_marginals=model_for_marg,
                marg_regularization = [(e_X, X, p_X), (e_Y, Y, q_Y)],
                λ_marg_reg=λ_marg,
                kwargs...)
    end
end

function Wasserstein_Barycenter(model::SMT.PSDModelOrthonormal{d2, T},
                        measures::AbstractVector{<:Function},
                        weights::AbstractVector{T},
                        ϵ::T,
                        XY::PSDDataVector{T};
                        X=nothing, Y=nothing,
                        preconditioner::Union{<:SMT.ConditionalMapping{d2, 0, T}, Nothing}=nothing,
                        reference::Union{<:SMT.ReferenceMap{d2, 0, T}, Nothing}=nothing,
                        use_putinar=true,
                        use_preconditioner_cost=false,
                        λ_marg=nothing,
                        kwargs...
            ) where {d2, T<:Number}
    d = d2 * (length(measures) + 1)

    throw(error("Not implemented yet"))

end


end