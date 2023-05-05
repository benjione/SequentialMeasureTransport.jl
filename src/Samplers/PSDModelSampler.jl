using Roots: find_zero

struct PSDModelSampler{d, T<:Number} <: Sampler{T}
    model::AbstractPSDModelOrthonormal{d, T} # model to sample from
    margins::Vector{<:AbstractPSDModelOrthonormal{<:Any, T}} # start with x_{≤1}, then x_{≤2}, ...
    integrals::Vector{TraceModel{T}} # integrals of marginals
    function PSDModelSampler(model::AbstractPSDModelOrthonormal{d, T}) where {d, T<:Number}
        model = normalize(model) # create normalized copy
        margins = [marginalize(model, collect(k:d)) for k in 2:d]
        margins = [margins; model] # add the full model as last
        integrals = map((x,k)->integral(x, k), margins, 1:d)
        new{d, T}(model, margins, integrals)
    end
end

Sampler(model::AbstractPSDModelOrthonormal) = PSDModelSampler(model)

function pushforward_u(sampler::PSDModelSampler{d, T}, u::PSDdata{T}) where {d, T<:Number}
    x = zeros(T, d)
    ## T^{-1}(x_1,...,x_k) functions, z=x_k
    f(k) = begin
        if k==1
            z->sampler.integrals[k](z) - u[k]
        else
            z->(sampler.integrals[k]([x[1:k-1]; z])/sampler.margins[k-1](x[1:k-1])) - u[k]
        end
    end
    for k=1:d
        left, right = domain_interval(sampler.model, k)
        x[k] = find_zero(f(k), (left, right))
    end
    return x
end

function sample(sampler::PSDModelSampler{d, T}) where {d, T<:Number}
    return pushforward_u(sampler, rand(T, d))
end


# function sample_subdomain(sampler::PSDModelSampler{d, T},
#                         marginal_points::PSDDataVector{T},
#                         dQ::Function, amount_points::Int;
#                         threading=false) where {d, T<:Number}
#     subdomain_check = subdomain_checker_in_u(sampler, marginal_points, dQ)
#     # generate random numbers in subdomain
#     u = rand(T, d, amount_points)
#     for i in 1:amount_points
#         while !subdomain_check(u[:,i])
#             u[:, i] = rand(T, d)
#         end
#     end
#     # pushforward to x space
#     if threading
#         u = slice_matrix(u)
#         x = similar(u)
#         Threads.@threads for i in 1:amount_points
#             x[i] = pushforward_u(sampler, u[i])
#         end
#         return x
#     else
#         return pushforward_u.(Ref(sampler), slice_matrix(u))
#     end
# end

function pullback_x(sampler::PSDModelSampler{d, T}, 
                        x::PSDdata{T}) where {d, T<:Number}
    f(k) = begin
        if k==1
            z->sampler.integrals[k](z)
        else
            z->(sampler.integrals[k]([x[1:k-1]; z])/sampler.margins[k-1](x[1:k-1]))
        end
    end
    u = similar(x)
    for k=1:d
        u[k] = f(k)(x[k])
    end
    return u
end

# convex subdomain checker, but if the subdomain is convex in 
# x space, it is not necessarily convex in u space
# function subdomain_checker_in_u(sampler::PSDModelSampler{d, T},
#                                 marginal_points::PSDDataVector{T},
#                                 dQ::Function) where {d, T<:Number}
#     @assert dQ(rand(d)) isa PSDdata
#     f(k) = begin
#         if k==1
#             z->sampler.integrals[k](z)
#         else
#             z->(sampler.integrals[k]([x[1:k-1]; z])/sampler.margins[k-1](x[1:k-1]))
#         end
#     end
#     marginal_points_u = similar(marginal_points)
#     for (i, x) in enumerate(marginal_points)
#         marginal_points_u[i] = pullback_x(sampler, x)
#     end
#     # calculate dQ(x) / dT(x) for all x in marginal_points
#     dT(x) = sampler.model(x)
#     scalar_product_points = dQ.(marginal_points) ./ dT.(marginal_points)

#     check_u_in_subdomain = let scalar_product_points=scalar_product_points, 
#                             marginal_points_u=marginal_points_u,
#                             marginal_points=marginal_points
#         u->begin
#             for i in 1:length(marginal_points)
#                 if dot(u - marginal_points_u[i], scalar_product_points[i]) > 0
#                     return false
#                 end
#             end
#             return true
#         end
#     end
#     return check_u_in_subdomain
# end