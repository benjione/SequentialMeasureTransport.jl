module OptimalTransport

using ..SequentialMeasureTransport
import ..SequentialMeasureTransport as SMT
using ..SequentialMeasureTransport: PSDDataVector
using Distributions
using FastGaussQuadrature: gausslegendre



function compute_Sinkhorn_distance(c, 
            smp::SMT.CondSampler{d}; 
            N=10000) where {d}
    _d = d ÷ 2
    X = rand(smp, N)
    return mapreduce(x -> c(x[1:_d], x[_d+1:end]), +, X) / N
end


function _left_pdf(sampler::SMT.ConditionalMapping{d}, x) where {d}
    p, w = gausslegendre(15)
    ## scale to [0, 1]
    p .= (p .+ 1) ./ 2
    w .= w ./ 2
    _d = d ÷ 2
    @assert length(x) == _d
    res = 0.0
    for k in Iterators.product([1:length(p) for _ in 1:_d]...)
        res += prod(w[[k...]]) .* pdf(sampler, [x; p[[k...]]])
    end
    return res
    # return sum(w .* [pdf(sampler, [x, y]) for y in p])
end


function _right_pdf(sampler::SMT.ConditionalMapping{d}, x) where {d}
    p, w = gausslegendre(15)
    ## scale to [0, 1]
    p .= (p .+ 1) ./ 2
    w .= w ./ 2
    _d = d ÷ 2
    @assert length(x) == _d
    res = 0.0
    for k in Iterators.product([1:length(p) for _ in 1:_d]...)
        res += prod(w[[k...]]) .* pdf(sampler, [p[[k...]]; x])
    end
    return res
    # return sum(w .* [pdf(sampler, [x, y]) for y in p])
end

function Barycentric_Projection_map(smp::SMT.CondSampler{d, dC};
                N=1000) where {d, dC}
    if dC == 0
        return Barycentric_Projection_map(smp, x->_left_pdf(smp, x); N=N)
    else
        ret_func = let smp=smp, N=N
            x -> sum(SMT.conditional_sample(smp, x, N, threading=true))[1] / N
        end
        return ret_func
    end
end

function Barycentric_Projection_map(smp::SMT.CondSampler{d},
                marginal::Function;
                N=1000) where {d}
    _d = d ÷ 2
    x_i, w_i = gausslegendre(15)
    x_i .= (x_i .+ 1) ./ 2
    w_i .= w_i ./ 2
    ret_func = let smp=smp, marginal=marginal, _d=_d, x_i=x_i, w_i=w_i
        x -> begin
            mx = marginal(x)
            sum(w_i .* x_i .* map(y->pdf(smp, [x; y]), x_i)) / mx
        end
    end
    return ret_func
end

function entropic_OT!(model::SMT.PSDModelOrthonormal{d2, T},
        cost::Function,
        p::Function,
        q::Function,
        ϵ::T,
        XY::PSDDataVector{T};
        X=nothing, Y=nothing,
        WX = nothing, WY = nothing,
        preconditioner::Union{<:SMT.ConditionalMapping{d2, <:Any, T}, Nothing}=nothing,
        reference::Union{<:SMT.ReferenceMap{d2, <:Any, T}, Nothing}=nothing,
        use_putinar=true,
        use_preconditioner_cost=false,
        λ_marg=nothing,
        marg_integration_mode=nothing,
        kwargs...) where {d2, T<:Number}
    @assert d2 % 2 == 0
    d = d2 ÷ 2


    if marg_integration_mode === nothing
        if preconditioner === nothing
            marg_integration_mode = :quadrature
        else
            marg_integration_mode = :conditional_MC
        end
    end

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
        λ_marg = 1.0*_order_rev_KL
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

    if WX === nothing
        WX = ones(T, length(X))
    end
    if WY === nothing
        WY = ones(T, length(Y))
    end
    
    ## evaluate the marginals on the original samples
    p_X = map(_p, X)
    q_Y = map(_q, Y)
    e_X = collect(1:d)
    e_Y = collect(d+1:d2)
    marg_reg = [(e_X, X, p_X, WX), (e_Y, Y, q_Y, WY)]
    marg_constr = nothing
    

    marg_cond_distr = if marg_integration_mode == :conditional_MC
        [(x, nr) -> SMT.conditional_sample(prec, x, nr, threading=true),
            (x, nr) -> SMT.marginal_sample(prec, nr, threading=true),]
    else
        nothing
    end
    

    if use_putinar && (typeof(model) <: SMT.PSDModelPolynomial)
        D, C = SMT.get_semialgebraic_domain_constraints(model)
        return SMT._OT_JuMP!(model, cost_pb, ϵ, XY, ξ; mat_list=D, coef_list=C, 
                model_for_marginals=model_for_marg,
                marg_regularization = marg_reg,
                marg_constraints = marg_constr,
                marg_conditional_distr=marg_cond_distr,
                λ_marg_reg=λ_marg,
                kwargs...)
    else
        return SMT._OT_JuMP!(model, cost_pb, ϵ, XY, ξ; 
                model_for_marginals=model_for_marg,
                marg_regularization = marg_reg,
                marg_constraints = marg_constr,
                marg_conditional_distr=marg_cond_distr,
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