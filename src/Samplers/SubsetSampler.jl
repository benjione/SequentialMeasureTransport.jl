"""
Takes another sampler which acts on the subdomain of x and adds identity maps for the other variables.
"""
struct SubsetSampler{d, d_sub, T<:Number, 
            R<:ReferenceMap{d, T}, 
            R_sub<:ReferenceMap{d_sub, T}
        } <: Sampler{d, T, R}
    sampler::Sampler{d_sub, T}
    X::AbstractMatrix{T}            # bases X of subspace, X ∈ R^{d x d2}
    P::AbstractMatrix{T}            # Projection so that P * y ∈ S and (I - P) * y = X' * y ∈ S^⟂
    P_tilde::AbstractMatrix{T}      # transform of y ∈ R^d to subspace α ∈ R^d2
    R_map::R                        # reference map from R^d to uniform on [0,1]^d
    R_map_sub::R_sub                # reference map from R^d2 to uniform on [0,1]^d2
    function SubsetSampler{d}(sampler::Sampler{d2, T}, 
            sampler_variables::AbstractVector{Int},
            R_map::R, R_map_sub::R2) where {d, d2, T<:Number, R, R2}
        @assert d2 < d
        @assert all(1<=k<=d for k in sampler_variables)
        X = sparse(sampler_variables, 
                   1:length(sampler_variables), 
                   ones(T, length(sampler_variables)))
        P_tilde = X'
        P = X * P_tilde
        new{d, d2, T, R, R2}(sampler, X, P, P_tilde, R_map, R_map_sub)
    end
    function SubsetSampler{d}(sampler::Sampler{d2, T}, 
                            X::AbstractMatrix{T},
                            P::AbstractMatrix{T},
                            P_tilde::AbstractMatrix{T},
                            R_map::R,
                            R_map_sub::R2
            ) where {d, d2, T<:Number, R, R2}
        @assert d2 < d
        @assert size(X, 1) == d
        @assert size(X, 2) == d2
        @assert size(P, 1) == d
        @assert size(P, 2) == d
        @assert size(P_tilde, 1) == d2
        @assert size(P_tilde, 2) == d
        new{d, d2, T, R, R2}(sampler, X, P, P_tilde, R_map, R_map_sub)
    end
    function SubsetSampler{d}(sampler::Sampler{d2, T}, 
                            X::AbstractMatrix{T},
                            R_map::R,
                            R_map_sub::R2
            ) where {d, d2, T<:Number, R, R2}
        @assert d2 < d
        @assert size(X, 1) == d
        @assert size(X, 2) == d2
        P_tilde = inv(X' * X) * X'
        P = X'
        new{d, d2, T, R, R2}(sampler, X, P, P_tilde, R_map, R_map_sub)
    end
end

"""
Using that V = V_1 ⊕ V_2, we can write the determinant as det(V) = det(V_1) * det(V_2).
Hence, det(∂v (P * T(v) + (1-P) v)) = det(∂v T(v)) * det(∂v v)
= det(∂v T(v)) * det(I) = det(∂v T(v)).

Total expression of forward is with T being the sampler:
R^{-1}(y) = X * R_sub^{-1}(T(R_sub(X' R^{-1}(x)))) + (I - P) * R^{-1}(x)
"""
function Distributions.pdf(sampler::SubsetSampler, x::PSDdata)
    xs = pullback(sampler.R_map, x) # from [0,1]^d to R^d
    sub_x = sampler.P_tilde * xs # from R^d to R^d2
    sub_x_maped = pushforward(sampler.R_map_sub, sub_x) # from R^d2 to [0,1]^d2
    sub_u = pullback(sampler.sampler, sub_x_maped) # still in [0,1]^d2
    us = sampler.X * pullback(sampler.R_map_sub, sub_u) + (I - sampler.P) * xs
    # u = pushforward(sampler.R_map, us)
    T_inv_jac_det = Distributions.pdf(sampler.sampler, sub_x_maped) * 
                    inverse_Jacobian(sampler.R_map_sub, sub_u) * 
                    Jacobian(sampler.R_map_sub, sub_x)

    return inverse_Jacobian(sampler.R_map, x) * T_inv_jac_det * Jacobian(sampler.R_map, us)
end

function RandomSubsetProjection(T::Type{<:Number}, d::Int, d2::Int)
    X = qr(rand(T, d, d)).Q[:, 1:d2]
    P_tilde = X'
    P = X * X'
    return X, P, P_tilde
end

function project_to_subset(P_tilde::AbstractMatrix{T}, 
                R::ReferenceMap{d, T}, 
                R_sub::ReferenceMap{d2, T},
                x::PSDdata{T}) where {d, d2, T<:Number}
    us = pullback(R, x)
    sub_u = P_tilde * us
    sub_x = pushforward(R_sub, sub_u)
    return sub_x
end

function pushforward(sampler::SubsetSampler{<:Any, <:Any, T, R, R_sub}, 
                    u::PSDdata{T}
            ) where {T<:Number, R, R_sub}
    us = pullback(sampler.R_map, u)
    sub_u = sampler.P_tilde * us
    sub_x = pushforward(sampler.sampler, pushforward(sampler.R_map_sub, sub_u))
    xs = sampler.X * pullback(sampler.R_map_sub, sub_x) + (I - sampler.P) * us
    x = pushforward(sampler.R_map, xs)
    return x
end

function pullback(sampler::SubsetSampler{<:Any, <:Any, T, R}, x::PSDdata{T}) where {T<:Number, R}
    xs = pullback(sampler.R_map, x)
    sub_x = sampler.P_tilde * xs
    sub_u = pullback(sampler.sampler, pushforward(sampler.R_map_sub, sub_x))
    us = sampler.X * pullback(sampler.R_map_sub, sub_u) + (I - sampler.P) * xs
    u = pushforward(sampler.R_map, us)
    return u
end