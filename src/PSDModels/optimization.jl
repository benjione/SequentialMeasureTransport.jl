
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
    return loss(a.(X))
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
                   L_includes_normalization=false,
                   λ_1=0.0,
                   λ_2=1e-8,
                   trace=false,
                   pre_eval=true,
                   pre_eval_thresh=5000,
                   normalization_constraint=false,
                   kwargs...
            ) where {T<:Number}

    if (normalization_constraint || L_includes_normalization) && !(a isa PSDModelPolynomial)
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

    loss = let L=L, X=X
        if L_includes_normalization
            (A) -> begin
                return L([f_B(i, A) for i in 1:length(X)], tr(A))
            end
        else
            (A) -> begin
                return L([f_B(i, A) for i in 1:length(X)])
            end
        end
    end
    
    loss = let loss=loss
        if λ_1 == 0.0 && λ_2 == 0.0
            (A) -> loss(A)
        elseif λ_1 == 0.0
            (A) -> begin
                return loss(A) + λ_2 * opnorm(A, 2)^2
            end
        elseif λ_2 == 0.0
            (A) -> begin
                return loss(A) + λ_1 * nuclearnorm(A)
            end
        else
            (A) -> begin
                return loss(A) + λ_1 * nuclearnorm(A) + λ_2 * opnorm(A, 2)^2
            end
        end
    end

    solution = optimize_PSD_model(a.B, loss;
                                trace=trace,
                                normalization_constraint=normalization_constraint,
                                _filter_kwargs(kwargs, 
                                        _optimize_PSD_kwargs
                                )...)
    set_coefficients!(a, solution)
    return nothing
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
                min_IRLS_iter=3,
                stopping_criteria=1e-6,
                kwargs...
            ) where {T<:Number}
    weights = ones(T, length(X))
    fit!(a, X, Y, weights; kwargs...)
    B = a.B
    for iter in 1:max_IRLS_iter
        weights = reweight.(a.(X, Ref(B)))
        fit!(a, X, Y, weights; kwargs...)
        if (iter > min_IRLS_iter) && norm(a.B - B) < stopping_criteria
            break
        end
        B = a.B
    end
    return nothing
end

function greedy_IRLS(a::PSDModelOrthonormal{d, T},  
                X::PSDDataVector{T},
                Y::Vector{T},
                reweight::Function,
                loss::Function;
                max_greedy_iterations=50,
                greedy_trace=false,
                max_order_difference=5,
                kwargs...
            ) where {d, T<:Number}
    IRLS!(a, X, Y, reweight; kwargs...)

    N = size(a.B, 1)
    current_loss = loss(a.(X))
    # Now we need to find the best orthonormal basis
    prop = next_index_proposals(a)
    d_loss = Dict{Vector{Int}, Real}()
    d_vec = Dict{Vector{Int}, Vector{Float64}}()

    for p in prop
        b = create_proposal(a, p)
        IRLS!(b, X, Y, reweight; fixed_variables=(1:N, 1:N),
                kwargs...)
        d_loss[p] = loss(b.(X))
        d_vec[p] = b.B[end, :]
    end

    _condition_to_continue(current_loss, d, iteration) = begin
        if iteration > max_greedy_iterations
            @info "Iteration $(iteration), is termination."
            @info "losses are $(d_loss)"
            @info "current loss is $(current_loss)"
            return false
        # elseif maximum(current_loss .- values(d)) < current_loss * (1.0-0.9)
        #     @info "Iteration $(iteration), is termination."
        #     @info "losses are $(d_loss)"
        #     @info "current loss is $(current_loss)"
        #     return true
        else
            return true
        end
    end

    iteration = 1
    ## 90 % of the current loss
    while _condition_to_continue(current_loss, d_loss, iteration)
        iteration += 1

        possible_keys = [x for x in keys(d_loss)]
        min_order = minimum([sum(x) for x in possible_keys])
        filter!(x->sum(x)<min_order+max_order_difference, possible_keys)
        key = reduce((x, y) -> d_loss[x] ≤ d_loss[y] ? x : y, possible_keys)

        if greedy_trace==true
            @info "Iteration $(iteration), add $(key) to the model."
            @info "losses are $(d_loss)"
            @info "current loss is $(current_loss)"
        end
        a = create_proposal(a, key, d_vec[key])
        IRLS!(a, X, Y, reweight;         
                kwargs...)

        delete!(d_loss, key)
        delete!(d_vec, key)
        
        current_loss = loss(a.(X))

        N = size(a.B, 1)
        
        prop = next_index_proposals(a)
        min_order = minimum([sum(x) for x in prop])
        filter!(x->sum(x)<min_order+max_order_difference, prop)
        # update dictionary
        for p in prop
            b = create_proposal(a, p)
            IRLS!(b, X, Y, reweight; 
                    fixed_variables=(1:N, 1:N),
                    kwargs...)
            if loss(b.(X)) > current_loss
                IRLS!(b, X, Y, reweight;
                    kwargs...)
            end
            d_loss[p] = loss(b.(X))
            d_vec[p] = b.B[end, :]
        end
    end

    return a
end

function greedy_fit(a::PSDModelOrthonormal{d, T},  
                X::PSDDataVector{T},
                Y::Vector{T},
                weights::Vector{T};
                max_greedy_iterations=50,
                greedy_trace=false,
                max_order_difference=5,
                greedy_convergence_tol=0.9,
                greedy_convergence_relaxation=2,
                kwargs...
            ) where {d, T<:Number}
    current_loss = fit!(a, X, Y, weights; kwargs...)
    loss_list = [current_loss]

    N = size(a.B, 1)
    # Now we need to find the best orthonormal basis
    prop = next_index_proposals(a)
    d_loss = Dict{Vector{Int}, Real}()
    d_vec = Dict{Vector{Int}, Vector{Float64}}()

    for p in prop
        b = create_proposal(a, p)
        d_loss[p] = fit!(b, X, Y, weights; fixed_variables=(1:N, 1:N),
                kwargs...)
        d_vec[p] = b.B[end, :]
    end

    _condition_to_continue(current_loss, d, iteration) = begin
        if iteration > max_greedy_iterations
            @info "Iteration $(iteration), is termination."
            @info "losses are $(d_loss)"
            @info "current loss is $(current_loss)"
            return false
        elseif !(minimum(values(d)) < current_loss * greedy_convergence_tol)
            if term_iteration < greedy_convergence_relaxation
                term_iteration += 1
                return true
            else
                @info "Iteration $(iteration), is termination."
                @info "losses are $(d_loss)"
                @info "current loss is $(current_loss)"
                return false
            end
        else
            term_iteration = 0
            return true
        end
    end

    iteration = 1
    ## 90 % of the current loss
    while _condition_to_continue(current_loss, d_loss, iteration)
        iteration += 1

        possible_keys = [x for x in keys(d_loss)]
        min_order = minimum([sum(x) for x in possible_keys])
        filter!(x->sum(x)<min_order+max_order_difference, possible_keys)
        key = reduce((x, y) -> d_loss[x] ≤ d_loss[y] ? x : y, possible_keys)

        if greedy_trace==true
            @info "Iteration $(iteration), add $(key) to the model."
            @info "losses are $(d_loss)"
            @info "current loss is $(current_loss)"
        end
        a = create_proposal(a, key, d_vec[key])
        current_loss = fit!(a, X, Y, weights;         
                kwargs...)
        
        push!(loss_list, current_loss)

        delete!(d_loss, key)
        delete!(d_vec, key)

        N = size(a.B, 1)
        
        prop = next_index_proposals(a)
        min_order = minimum([sum(x) for x in prop])
        filter!(x->sum(x)<min_order+max_order_difference, prop)
        # update dictionary
        for p in prop
            b = create_proposal(a, p)
            loss = fit!(b, X, Y, weights; 
                    fixed_variables=(1:N, 1:N),
                    kwargs...)
            # if loss > current_loss
            #     loss = fit!(b, X, Y, weights;
            #         kwargs...)
            # end
            d_loss[p] = loss
            d_vec[p] = b.B[end, :]
        end
    end

    return a, loss_list
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