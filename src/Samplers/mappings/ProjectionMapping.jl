"""
The projection mapping is always defined on the uniform distribution on [0,1]^d.
"""
struct ProjectionMapping{d, dC, T<:Number,
            dsub, dCsub,
            Mtype<:ConditionalMapping{dsub, dCsub, T},
            R<:ReferenceMap{d, dC, T}, R2<:ReferenceMap{dsub, dCsub, T}
        } <: SubsetMapping{d, dC, T, dsub, dCsub}
    sampler::Mtype
    X::AbstractMatrix{T}            # bases X of subspace, X ∈ R^{d x d2}
    P::AbstractMatrix{T}            # Projection so that P * y ∈ S and (I - P) * y = X' * y ∈ S^⟂
    P_tilde::AbstractMatrix{T}      # transform of y ∈ R^d to subspace α ∈ R^d2
    R_map::R                        # reference map from R^d to uniform on [0,1]^d
    R_map_sub::R2                # reference map from R^d2 to uniform on [0,1]^d2
    function ProjectionMapping{d, dC}(mapping::ConditionalMapping{dsub, dCsub, T}, 
            map_variables::AbstractVector{Int},
            R_map::R1, R_map_sub::R2) where {d, dC, dsub, dCsub, T<:Number, R1, R2}
        @assert dsub < d
        @assert all(1<=k<=d for k in map_variables)
        @assert dCsub ≤ dC
        if dC > 0
            @assert dCsub > 0
        end
        X = spzeros(T, d, dsub)
        for (i, j) in enumerate(map_variables)
            X[j, i] = 1
        end
        P_tilde = X'
        P = X * P_tilde
        new{d, dC, T, dsub, dCsub, typeof(mapping), R1, R2}(
                mapping, X, P, 
                P_tilde, R_map, 
                R_map_sub
            )
    end
    function ProjectionMapping{d}(mapping::Mapping{dsub, T}, 
            sampler_variables::AbstractVector{Int},
            R_map::R, R_map_sub::R2) where {d, dsub, T<:Number, R, R2}
            ProjectionMapping{d, 0}(mapping, sampler_variables, R_map, R_map_sub)
    end
    function ProjectionMapping{d}(sampler::Mapping{dsub, T}, 
                            X::AbstractMatrix{T},
                            P::AbstractMatrix{T},
                            P_tilde::AbstractMatrix{T},
                            R_map::R,
                            R_map_sub::R2
            ) where {d, dsub, T<:Number, R, R2}
        ProjectionMapping{d, 0}(sampler, X, P, P_tilde, R_map, R_map_sub)
    end
    function ProjectionMapping{d, dC}(mapping::ConditionalMapping{dsub, dCsub, T}, 
                            X::AbstractMatrix{T},
                            P::AbstractMatrix{T},
                            P_tilde::AbstractMatrix{T},
                            R_map::R,
                            R_map_sub::R2
            ) where {d, dC,dCsub, dsub, T<:Number, R, R2}
        @assert dsub < d
        @assert dCsub ≤ dC
        @assert size(X, 1) == d
        @assert size(X, 2) == dsub
        @assert size(P, 1) == d
        @assert size(P, 2) == d
        @assert size(P_tilde, 1) == dsub
        @assert size(P_tilde, 2) == d
        new{d, dC, T, dsub, dCsub, typeof(mapping), R, R2}(mapping, X, P, P_tilde, R_map, R_map_sub)
    end
end

"""
Using that V = V_1 ⊕ V_2, we can write the determinant as det(V) = det(V_1) * det(V_2).
Hence, det(∂v (P * T(v) + (1-P) v)) = det(∂v T(v)) * det(∂v v)
= det(∂v T(v)) * det(I) = det(∂v T(v)).

Total expression of forward is with T being the sampler:
R^{-1}(y) = X * R_sub^{-1}(T(R_sub(X' R^{-1}(x)))) + (I - P) * R^{-1}(x)
"""


function inverse_Jacobian(sampler::ProjectionMapping{<:Any, <:Any, T, R, R_sub, <:Any, <:Any, ST}, 
                x::PSDdata{T}
            ) where {T<:Number, R, R_sub, ST}
    xs = pullback(sampler.R_map, x) # from [0,1]^d to R^d
    sub_x = sampler.P_tilde * xs # from R^d to R^d2
    sub_x_maped = pushforward(sampler.R_map_sub, sub_x) # from R^d2 to [0,1]^d2
    sub_u = pullback(sampler.sampler, sub_x_maped) # still in [0,1]^d2
    us = sampler.X * pullback(sampler.R_map_sub, sub_u) + (I - sampler.P) * xs
    # u = pushforward(sampler.R_map, us)
    T_inv_jac_det = inverse_Jacobian(sampler.sampler, sub_x_maped) * 
                    inverse_Jacobian(sampler.R_map_sub, sub_u) * 
                    Jacobian(sampler.R_map_sub, sub_x)

    return inverse_Jacobian(sampler.R_map, x) * T_inv_jac_det * Jacobian(sampler.R_map, us)
end
Jacobian(sampler::ProjectionMapping{<:Any, <:Any, T, R, R_sub, <:Any, <:Any, ST}, 
                x::PSDdata{T}
            ) where {T<:Number, R, R_sub, ST} = 1.0/inverse_Jacobian(sampler, pushforward(sampler, x))

function inverse_log_Jacobian(sampler::ProjectionMapping{<:Any, <:Any, T, R, R_sub, <:Any, <:Any, ST}, 
                x::PSDdata{T}
            ) where {T<:Number, R, R_sub, ST}
    xs = pullback(sampler.R_map, x) # from [0,1]^d to R^d
    sub_x = sampler.P_tilde * xs # from R^d to R^d2
    sub_x_maped = pushforward(sampler.R_map_sub, sub_x) # from R^d2 to [0,1]^d2
    sub_u = pullback(sampler.sampler, sub_x_maped) # still in [0,1]^d2
    us = sampler.X * pullback(sampler.R_map_sub, sub_u) + (I - sampler.P) * xs
    # u = pushforward(sampler.R_map, us)
    T_inv_jac_det = inverse_log_Jacobian(sampler.sampler, sub_x_maped) +
                    inverse_log_Jacobian(sampler.R_map_sub, sub_u) +
                    log_Jacobian(sampler.R_map_sub, sub_x)

    return inverse_log_Jacobian(sampler.R_map, x) + T_inv_jac_det + log_Jacobian(sampler.R_map, us)
end
function log_Jacobian(sampler::ProjectionMapping{<:Any, <:Any, T, R, R_sub, <:Any, <:Any, ST}, 
                x::PSDdata{T}
            ) where {T<:Number, R, R_sub, ST}
    return -inverse_log_Jacobian(sampler, pushforward(sampler, x))
end

###
## Helper functions
@inline _d_sub_marg(
            sampler::ProjectionMapping{<:Any, <:Any, <:Any, dsub, dCsub}
        ) where {dsub, dCsub} = dsub - dCsub
@inline _marg_P(
            sampler::ProjectionMapping{d, dC, T, dsub, dCsub, ST}
        ) where {d, dC, T, dsub, dCsub, ST} = sampler.P[1:_d_marg(sampler), 1:_d_marg(sampler)]
@inline _marg_P_tilde(
            sampler::ProjectionMapping{d, dC, T, dsub, dCsub, ST}
        ) where {d, dC, T, dsub, dCsub, ST} = sampler.P_tilde[1:_d_sub_marg(sampler), 1:_d_marg(sampler)]
@inline _marg_X(
            sampler::ProjectionMapping{d, dC, T, dsub, dCsub, ST}
        ) where {d, dC, T, dsub, dCsub, ST} = sampler.X[1:_d_marg(sampler), 1:_d_sub_marg(sampler)]


function RandomSubsetProjection(T::Type{<:Number}, d::Int, dsub::Int)
    # X = qr(randn(T, d, d)).Q[:, 1:dsub]
    X = rand(Stiefel(d, dsub)) .|> T
    P_tilde = X'
    P = X * X'
    return X, P, P_tilde
end

function RandomConditionalSubsetProjection(T::Type{<:Number}, 
                                d::Int, 
                                dC::Int, 
                                dsub::Int, 
                                dCsub::Int
                            )
    dx = d - dC
    d_marg_red = dsub - dCsub
    @assert dCsub ≤ dC
    @assert d_marg_red ≤ dx
    Xm = rand(Stiefel(dx, d_marg_red)) .|> T
    Xc = rand(Stiefel(dC, dCsub)) .|> T
    # Xm = qr(randn(T, dx, dx)).Q[:, 1:d_marg_red]
    # Xc = qr(randn(T, dC, dC)).Q[:, 1:dCsub]
    X = zeros(T, d, dsub)
    X[1:dx, 1:d_marg_red] = Xm
    X[dx+1:d, d_marg_red+1:dsub] = Xc
    P_tilde = X'
    P = X * X'
    return X, P, P_tilde
end

function project_to_subset(P_tilde::AbstractMatrix{T}, 
                R::ReferenceMap{d, <:Any, T}, 
                R_sub::ReferenceMap{dsub, <:Any, T},
                x::PSDdata{T}) where {d, dsub, T<:Number}
    us = pullback(R, x)
    sub_u = P_tilde * us
    sub_x = pushforward(R_sub, sub_u)
    return sub_x
end

function pushforward(sampler::ProjectionMapping{<:Any, <:Any, T, <:Any, <:Any, ST, R, R_sub}, 
                    u::PSDdata{T}
            ) where {T<:Number, R, R_sub, ST}
    us = pullback(sampler.R_map, u)
    sub_u = sampler.P_tilde * us
    sub_x = pushforward(sampler.sampler, pushforward(sampler.R_map_sub, sub_u))
    xs = sampler.X * pullback(sampler.R_map_sub, sub_x) + (I - sampler.P) * us
    x = pushforward(sampler.R_map, xs)
    return x
end

function pullback(sampler::ProjectionMapping{<:Any, <:Any, T, <:Any, <:Any, ST, R, R2}, 
                    x::PSDdata{T}
            ) where {T<:Number, R, R2, ST}
    xs = pullback(sampler.R_map, x)
    sub_x = sampler.P_tilde * xs
    sub_u = pullback(sampler.sampler, pushforward(sampler.R_map_sub, sub_x))
    us = sampler.X * pullback(sampler.R_map_sub, sub_u) + (I - sampler.P) * xs
    u = pushforward(sampler.R_map, us)
    return u
end


########################
# ConditionalSampler
########################

function marginal_inverse_Jacobian(sampler::ProjectionMapping{d, dC, T, dsub, dCsub, ST, R1, R2}, 
                x::PSDdata{T}
            ) where {d, dC, T<:Number, dsub, dCsub, ST, R1, R2}
    xs = marginal_pullback(sampler.R_map, x) # from [0,1]^dx to R^dx
    sub_x = _marg_P_tilde(sampler) * xs # from R^dx to R^d_sub_marg
    sub_x_maped = marginal_pushforward(sampler.R_map_sub, sub_x) # from R^d_sub_marg to [0,1]^d_sub_marg
    sub_u = marginal_pullback(sampler.sampler, sub_x_maped) # still in [0,1]^d_sub_marg
    us = _marg_X(sampler) * marginal_pullback(sampler.R_map_sub, sub_u) + (I - _marg_P(sampler)) * xs
    # u = pushforward(sampler.R_map, us)
    T_inv_jac_det = marginal_inverse_Jacobian(sampler.sampler, sub_x_maped) * 
                    marginal_inverse_Jacobian(sampler.R_map_sub, sub_u) * 
                    marginal_Jacobian(sampler.R_map_sub, sub_x)

    return marginal_inverse_Jacobian(sampler.R_map, x) * T_inv_jac_det * marginal_Jacobian(sampler.R_map, us)
end

function marginal_inverse_log_Jacobian(sampler::ProjectionMapping{d, dC, T, dsub, dCsub, ST, R1, R2}, 
                x::PSDdata{T}
            ) where {d, dC, T<:Number, dsub, dCsub, ST, R1, R2}
    xs = marginal_pullback(sampler.R_map, x) # from [0,1]^dx to R^dx
    sub_x = _marg_P_tilde(sampler) * xs # from R^dx to R^d_sub_marg
    sub_x_maped = marginal_pushforward(sampler.R_map_sub, sub_x) # from R^d_sub_marg to [0,1]^d_sub_marg
    sub_u = marginal_pullback(sampler.sampler, sub_x_maped) # still in [0,1]^d_sub_marg
    us = _marg_X(sampler) * marginal_pullback(sampler.R_map_sub, sub_u) + (I - _marg_P(sampler)) * xs
    # u = pushforward(sampler.R_map, us)
    T_inv_jac_det = marginal_inverse_log_Jacobian(sampler.sampler, sub_x_maped) + 
                    marginal_inverse_log_Jacobian(sampler.R_map_sub, sub_u) +
                    marginal_log_Jacobian(sampler.R_map_sub, sub_x)

    return marginal_inverse_log_Jacobian(sampler.R_map, x) + T_inv_jac_det + marginal_log_Jacobian(sampler.R_map, us)
end

function marginal_Jacobian(sampler::ProjectionMapping{d, dC, T, dsub, dCsub, ST, R1, R2}, 
                x::PSDdata{T}
            ) where {d, dC, T<:Number, dsub, dCsub, ST, R1, R2}
    return 1.0/marginal_inverse_Jacobian(sampler, marginal_pushforward(sampler, x))
end

function marginal_log_Jacobian(sampler::ProjectionMapping{d, dC, T, dsub, dCsub, ST, R1, R2}, 
                x::PSDdata{T}
            ) where {d, dC, T<:Number, dsub, dCsub, ST, R1, R2}
    return -marginal_inverse_log_Jacobian(sampler, marginal_pushforward(sampler, x))
end

function marginal_pushforward(sampler::ProjectionMapping{d, dC, T, dsub, dCsub, ST, R1, R2}, 
                u::PSDdata{T}
            ) where {d, dC, T<:Number, dsub, dCsub, ST, R1, R2}
    us = marginal_pullback(sampler.R_map, u)
    sub_u = _marg_P_tilde(sampler) * us
    sub_x = marginal_pushforward(sampler.sampler, marginal_pushforward(sampler.R_map_sub, sub_u))
    xs = _marg_X(sampler) * marginal_pullback(sampler.R_map_sub, sub_x) + (I - _marg_P(sampler)) * us
    x = marginal_pushforward(sampler.R_map, xs)
    return x
end

function marginal_pullback(sampler::ProjectionMapping{d, dC, T, dsub, dCsub, ST, R1, R2},
                x::PSDdata{T}
            ) where {d, dC, T<:Number, dsub, dCsub, ST, R1, R2}
    xs = marginal_pullback(sampler.R_map, x)
    sub_x = _marg_P_tilde(sampler) * xs
    sub_u = marginal_pullback(sampler.sampler, marginal_pushforward(sampler.R_map_sub, sub_x))
    us = _marg_X(sampler) * marginal_pullback(sampler.R_map_sub, sub_u) + (I - _marg_P(sampler)) * xs
    u = marginal_pushforward(sampler.R_map, us)
    return u
end
