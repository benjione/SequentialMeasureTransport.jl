
fit!(a::PSDModel, 
        X::PSDDataVector{T}, 
        Y::Vector{T}; 
        kwargs...
    ) where {T<:Number} = fit!(a, X, Y, ones(T, length(X)); kwargs...)
function fit!(a::PSDModel{T}, 
                X::PSDDataVector{T}, 
                Y::Vector{T},
                weights::Vector{T}; 
                kwargs...
            ) where {T<:Number}
    N = length(X)
    loss(Z) = (1.0/N) * sum((Z .- Y).^2 .* weights)

    minimize!(a, loss, X; kwargs...)
    return nothing
end

"""
minimize!(a::PSDModel{T}, L::Function, X::PSDDataVector{T}; λ_1=1e-8,
                    trace=false,
                    maxit=5000,
                    tol=1e-6,
                    pre_eval=true,
                    pre_eval_thresh=5000,
                ) where {T<:Number}

Minimizes ``B^* = \\argmin_B L(a_B(x_1), a_B(x_2), ...) + λ_1 tr(B) `` and returns the modified PSDModel with the right matrix B.
"""
function minimize!(a::PSDModel{T}, 
                   L::Function, 
                   X::PSDDataVector{T};
                   λ_1=0.0,
                   λ_2=1e-8,
                   trace=false,
                   pre_eval=true,
                   pre_eval_thresh=5000,
                   normalization_constraint=false,
                   kwargs...
            ) where {T<:Number}

    if normalization_constraint && !(a isa PSDModelPolynomial)
        @error "Normalization constraint only implemented for tensorized polynomial model!"
        return nothing
    end
    N = length(X)
    f_B = if pre_eval && (N < pre_eval_thresh)
        let K = reduce(hcat, Φ.(Ref(a), X))
            (i, A) -> begin
                v = K[:, i]
                return dot(v, A, v)
            end
        end
    else
        (i, A) -> begin
            return a(X[i], A)
        end
    end
    loss =
        if λ_1 == 0.0 && λ_2 == 0.0
            (A) -> begin
                return L([f_B(i, A) for i in 1:length(X)])
            end
        elseif λ_1 == 0.0
            (A) -> begin
                return L([f_B(i, A) for i in 1:length(X)]) + λ_2 * opnorm(A, 2)^2
            end
        elseif λ_2 == 0.0
            (A) -> begin
                return L([f_B(i, A) for i in 1:length(X)]) + λ_1 * nuclearnorm(A)
            end
        else
            (A) -> begin
                return L([f_B(i, A) for i in 1:length(X)]) + λ_1 * nuclearnorm(A) + λ_2 * opnorm(A, 2)^2
            end
        end

    solution = optimize_PSD_model(a.B, loss;
                                trace=trace,
                                normalization_constraint=normalization_constraint,
                                _filter_kwargs(kwargs, 
                                        _optimize_PSD_kwargs
                                )...)
    set_coefficients!(a, solution)
end


"""
IRLS!(a::PSDModel{T},  
        X::PSDDataVector{T},
        Y::Vector{T},
        reweight::Function;
        λ_1=1e-8,
        trace=false,
        maxit=5000,
        tol=1e-6,
        pre_eval=true,
        pre_eval_thresh=5000,
    ) where {T<:Number}

Minimizes ``B^* = \\argmin_B L(a_B(x_1), a_B(x_2), ...) + λ_1 tr(B) `` and returns the modified PSDModel with the right matrix B.
"""
function IRLS!(a::PSDModel{T},  
                X::PSDDataVector{T},
                Y::Vector{T},
                reweight::Function;
                max_IRLS_iter=10,
                kwargs...
            ) where {T<:Number}
    weights = ones(T, length(X))
    fit!(a, X, Y, weights; kwargs...)
    B = a.B
    for _ in 1:max_IRLS_iter
        weights = reweight.(a.(X, Ref(B)))
        fit!(a, X, Y, weights; kwargs...)
        if norm(a.B - B) < 1e-6
            break
        end
        B = a.B
    end
    return nothing
end
# function IRLS!(a::PSDModel{T},  
#                 X::PSDDataVector{T},
#                 Y::Vector{T},
#                 reweight::Function;
#                 λ_1=0.0,
#                 λ_2=1e-8,
#                 trace=false,
#                 pre_eval=true,
#                 pre_eval_thresh=5000,
#                 normalization_constraint=false,
#                 kwargs...
#             ) where {T<:Number}

#     if normalization_constraint && !(a isa PSDModelPolynomial)
#         @error "Normalization constraint only implemented for tensorized polynomial model!"
#         return nothing
#     end
#     N = length(X)
#     f_B = if pre_eval && (N < pre_eval_thresh)
#         let K = reduce(hcat, Φ.(Ref(a), X))
#             (i, A) -> begin
#                 v = K[:, i]
#                 return dot(v, A, v)
#             end
#         end
#     else
#         (i, A) -> begin
#             return a(X[i], A)
#         end
#     end

#     L(Z, Z_old) = (1.0/N) * mapreduce(
#         (z,y,zold)->(z - y)^2 * reweight(zold), +, Z, Y, Z_old)

#     loss =
#         if λ_1 == 0.0 && λ_2 == 0.0
#             (A, B) -> begin
#                 return L([f_B(i, A) for i in 1:length(X)],
#                         [f_B(i, B) for i in 1:length(X)])
#             end
#         elseif λ_1 == 0.0
#             (A, B) -> begin
#                 return L([f_B(i, A) for i in 1:length(X)],
#                         [f_B(i, B) for i in 1:length(X)]) + 
#                         λ_2 * opnorm(A, 2)^2
#             end
#         elseif λ_2 == 0.0
#             (A, B) -> begin
#                 return L([f_B(i, A) for i in 1:length(X)],
#                         [f_B(i, B) for i in 1:length(X)]) + 
#                         λ_1 * nuclearnorm(A)
#             end
#         else
#             (A, B) -> begin
#                 return L([f_B(i, A) for i in 1:length(X)],
#                         [f_B(i, B) for i in 1:length(X)]) + 
#                         λ_1 * nuclearnorm(A) + 
#                         λ_2 * opnorm(A, 2)^2
#             end
#         end
    
#     # change that
#     convergence(A, B, k) = begin
#         if k>10
#             return true
#         else
#             return false
#         end
#     end

#     solution = iteratively_optimize_convex(a.B, loss,
#                                 convergence;
#                                 trace=trace,
#                                 normalization_constraint=normalization_constraint,
#                                 _filter_kwargs(kwargs, 
#                                         _optimize_PSD_kwargs
#                                 )...)
#     set_coefficients!(a, solution)
# end