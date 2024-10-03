import JuMP

struct JuMPOptProp{T} <: OptProp{T}
    initial::AbstractMatrix{T}
    loss::Function
    normalization::Bool           # if tr(X) = 1
    optimizer
    fixed_variables
    trace::Bool
    marg_constraints          # (expr, result)
    function JuMPOptProp(
            initial::AbstractMatrix{T}, 
            loss::Function;
            trace=false,
            optimizer=nothing,
            fixed_variables=nothing,
            normalization=false,
            marg_constraints = nothing,
        ) where {T<:Number}
        if optimizer === nothing
            optimizer = con.MOI.OptimizerWithAttributes(
                Hypatia.Optimizer,
            )
        else
            @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
        end
        new{T}(initial, 
                loss,
                normalization,
                optimizer,
                fixed_variables,
                trace,
                marg_constraints
            )
    end
end


function optimize(prob::JuMPOptProp{T}) where {T<:Number}
    verbose_solver = prob.trace ? true : false

    model = JuMP.Model(prob.optimizer)
    if verbose_solver
        JuMP.unset_silent(model)
    else
        JuMP.set_silent(model)
    end
    N = size(prob.initial, 1)
    JuMP.@variable(model, B[1:N, 1:N], PSD)

    JuMP.set_start_value.(B, prob.initial)
    if prob.fixed_variables !== nothing
        @info "some variables are fixed"
        JuMP.fix.(B[prob.fixed_variables], prob.initial[prob.fixed_variables], force=true)
        # JuMP.@constraint(model, B[prob.fixed_variables] .== prob.initial[prob.fixed_variables])
    end
    JuMP.@objective(model, Min, prob.loss(B))

    if prob.normalization
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        JuMP.@constraint(model, tr(B) == 1)
    end

    if prob.marg_constraints !== nothing
        for (marg_model, B_marg) in prob.marg_constraints
            @info "fixing marginal"
            JuMP.@constraint(model, Hermitian(marg_model.P * (marg_model.M .* B) * marg_model.P') == B_marg)
        end
    end

    @show model

    JuMP.optimize!(model)
    res_B = Hermitian(T.(JuMP.value(B)))

    finalize(model)
    model = nothing
    GC.gc()
    return res_B
end

_least_squares_JuMP!(a::PSDModel{T}, X::PSDDataVector{T}, Y::Vector{T}; kwargs...) where {T<:Number} = 
                        _least_squares_JuMP!(a, X, Y, ones(T, length(Y)); kwargs...)

function _least_squares_JuMP!(a::PSDModel{T},
                    X::PSDDataVector{T},
                    Y::Vector{T},
                    W::Vector{T};
                    λ_1 = 0.0,
                    λ_2 = 0.0,
                    trace=false,
                    optimizer=nothing,mat_list = nothing,
                    coef_list = nothing,
                    set_start_value = true) where {T<:Number}
    verbose_solver = trace ? true : false

    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            Hypatia.Optimizer,
        )
    else
        @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
    end

    model = JuMP.Model(optimizer)
    JuMP.set_string_names_on_creation(model, false)
    if verbose_solver
        JuMP.unset_silent(model)
    else
        JuMP.set_silent(model)
    end
    
    function create_M(PSD_model, x)
        m = length(x)
        n = size(PSD_model.B)[1]
        n = (n*(n+1))÷2
        M = zeros(m, n)
        for i = 1:m
            v = Φ(PSD_model, x[i])
            M_i = v*v'
            M_i = M_i + M_i' - Diagonal(diag(M_i))
            M_i = Hermitian_to_low_vec(M_i)
            M[i, :] = M_i
        end
        return M
    end

    trace && print("Create M....")
    t = time()
    M = create_M(a, X)
    dt = time() - t
    trace && print("done! - $(dt)  \n")


    N = size(a.B, 1)
    JuMP.@variable(model, B[1:N, 1:N], PSD)

    if mat_list !== nothing
        JuMP.@variable(model, D[1:length(mat_list), 1:N, 1:N])
        for i=1:length(mat_list)
            JuMP.@constraint(model, D[i,:,:] in JuMP.PSDCone())
        end
        JuMP.@expression(model, SoS_comb, sum(
                                            sum(
                                                coef_list[i][k] * mat_list[i][k] * D[i,:,:] * mat_list[i][k]'
                                            for k=1:length(coef_list[i])) 
                                        for i=1:length(mat_list)))
    end

    if set_start_value
        JuMP.set_start_value.(B, a.B)
    end
    B_red = Hermitian_to_low_vec(B)

    JuMP.@variable(model, t)
    size_cone = length(Y)

    W_sq_root = sqrt.(W)
    if mat_list !== nothing
        SoS_red = Hermitian_to_low_vec(SoS_comb)
        JuMP.@constraint(model, [t; W_sq_root .* (M * (B_red + SoS_red) - Y)] in JuMP.MOI.SecondOrderCone(size_cone+1))
    else
        JuMP.@constraint(model, [t; W_sq_root .* (M * B_red - Y)] in JuMP.MOI.SecondOrderCone(size_cone+1))
    end
    JuMP.@objective(model, Min, t)
    JuMP.optimize!(model)

    # res_B = Hermitian(T.(JuMP.value(B)))
    res_B = if mat_list === nothing
        res_B = Hermitian(T.(JuMP.value(B)))
        res_B
    else
        D = JuMP.value.(D)
        res_B1 = T.(JuMP.value(B))
        res_B2 = sum(
                    sum(
                        coef_list[i][k] * mat_list[i][k] * T.(D[i,:,:]) * mat_list[i][k]'
                    for k=1:length(coef_list[i])) 
                for i=1:length(mat_list))
        res_B = res_B1 + res_B2
        res_B
    end

    set_coefficients!(a, Hermitian(res_B))
    _loss(Z) = (1.0/length(Z)) * sum((Z .- Y).^2)

    finalize(model)
    model = nothing
    GC.gc()
    return _loss(a.(X))
end

# function _least_squares_JuMP!(a::PSDModel{T},
#                     X::PSDDataVector{T},
#                     Y::Vector{T};
#                     λ_1 = 0.0,
#                     λ_2 = 0.0,
#                     trace=false,
#                     optimizer=nothing,) where {T<:Number}
#     verbose_solver = trace ? true : false

#     if optimizer===nothing
#         optimizer = con.MOI.OptimizerWithAttributes(
#             Hypatia.Optimizer,
#         )
#     else
#         @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
#     end

#     model = JuMP.Model(optimizer)
#     JuMP.set_string_names_on_creation(model, false)
#     if verbose_solver
#         JuMP.unset_silent(model)
#     else
#         JuMP.set_silent(model)
#     end
    
#     function create_M(PSD_model, x)
#         m = length(x)
#         n = size(PSD_model.B)[1]
#         n = (n*(n+1))÷2
#         M = zeros(m, n)
#         for i = 1:m
#             v = Φ(PSD_model, x[i])
#             M_i = v*v'
#             M_i = M_i + M_i' - Diagonal(diag(M_i))
#             M_i = Hermitian_to_low_vec(M_i)
#             M[i, :] = M_i
#         end
#         return M
#     end

#     trace && print("Create M....")
#     t = time()
#     M = create_M(a, X)
#     dt = time() - t
#     trace && print("done! - $(dt)  \n")
#     M2 = M' * M
#     Y2 = M' * Y
#     # print("Condition number of M is ", cond(M),"\n")
#     # trace && print("Calculate non PSD estimate of B...")
#     # t = time()
#     # res_B_vec = M \ Y
#     # dt = time() - t
#     # trace && print("done! - $(dt) \n")
#     # res_B_vec = reshape(res_B_vec, length(res_B_vec))


#     N = size(a.B, 1)
#     JuMP.@variable(model, _B[1:N, 1:N], PSD)
#     if mat_list !== nothing
#         N_SoS_comb = [size(m[1], 2) for m in mat_list]
#         D = []
#         for i=1:length(mat_list)
#             _D = JuMP.@variable(model, [1:N_SoS_comb[i], 1:N_SoS_comb[i]], PSD)
#             push!(D, _D)
#         end
#         JuMP.@expression(model, SoS_comb, sum(
#                                             sum(
#                                                 coef_list[i][k] * mat_list[i][k] * D[i] * mat_list[i][k]'
#                                             for k=1:length(coef_list[i])) 
#                                         for i=1:length(mat_list)))
#     end

#     B = if mat_list !== nothing
#         _B + SoS_comb
#     else
#         _B
#     end

#     JuMP.set_start_value.(B, a.B)
#     B_red = Hermitian_to_low_vec(B)
#     JuMP.@variable(model, t)
#     size_cone = length(Y2)
 
#     JuMP.@constraint(model, [t; M2 * B_red - Y2] in JuMP.MOI.SecondOrderCone(size_cone+1))
#     JuMP.@objective(model, Min, t)
#     JuMP.optimize!(model)

#     res_B = if mat_list === nothing
#         res_B = Hermitian(T.(JuMP.value(_B)))
#         res_B
#     else
#         D = JuMP.value.(D)
#         res_B1 = T.(JuMP.value(_B))
#         res_B2 = sum(
#                     sum(
#                         coef_list[i][k] * mat_list[i][k] * T.(D[i]) * mat_list[i][k]'
#                     for k=1:length(coef_list[i])) 
#                 for i=1:length(mat_list))
#         res_B = res_B1 + res_B2
#         res_B
#     end
#     set_coefficients!(a, Hermitian(res_B))
#     _loss(Z) = (1.0/length(Z)) * sum((Z .- Y).^2)

#     finalize(model)
#     model = nothing
#     GC.gc()
#     return _loss(a.(X))
# end

function _closest_PSD_JuMP!(A::Hermitian{T}; optimizer=nothing,
            mat_list = nothing, coef_list = nothing,
    ) where {T<:Number}

    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            Hypatia.Optimizer,
        )
    else
        @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
    end

    model = JuMP.Model(optimizer)
    JuMP.set_string_names_on_creation(model, false)
    if verbose_solver
        JuMP.unset_silent(model)
    else
        JuMP.set_silent(model)
    end

    N = size(A, 1)
    JuMP.@variable(model, B[1:N, 1:N], PSD)

    if mat_list !== nothing
        N_SoS_comb = [size(m[1], 2) for m in mat_list]
        D = []
        for i=1:length(mat_list)
            _D = JuMP.@variable(model, [1:N_SoS_comb[i], 1:N_SoS_comb[i]], PSD)
            push!(D, _D)
        end
        JuMP.@expression(model, SoS_comb, sum(
                                            sum(
                                                coef_list[i][k] * mat_list[i][k] * D[i] * mat_list[i][k]'
                                            for k=1:length(coef_list[i])) 
                                        for i=1:length(mat_list)))
    end

    # JuMP.set_start_value.(B, A)
    # B_red = Hermitian_to_low_vec(B)
    JuMP.@variable(model, t)
    # size_cone = length(B_red)
    # A_vec = Hermitian_to_low_vec(A)
 
    # JuMP.@constraint(model, [t; B_red - res_B_vec] in JuMP.MOI.SecondOrderCone(size_cone+1))
    if mat_list !== nothing
        JuMP.@constraint(model, [t; vec(B + SoS_comb - A)] in JuMP.MOI.NormSpectralCone(N, N))
        # JuMP.@constraint(model, [t; vec(B + SoS_comb - A)] in JuMP.MOI.SecondOrderCone(N^2+1))
    else
        JuMP.@constraint(model, [t; vec(B - A)] in JuMP.MOI.NormSpectralCone(N, N))
    end
    JuMP.@objective(model, Min, t)
    JuMP.optimize!(model)

    if mat_list === nothing
        res_B = Hermitian(T.(JuMP.value(B)))
        return res_B
    else
        D = JuMP.value.(D)
        res_B1 = T.(JuMP.value(B))
        # res_B1[end,:] .= 0
        # res_B1[:,end] .= 0
        res_B2 = sum(
                    sum(
                        coef_list[i][k] * mat_list[i][k] * T.(D[i]) * mat_list[i][k]'
                    for k=1:length(coef_list[i])) 
                for i=1:length(mat_list))
        # res_B2[:,end-1:end] .= 0
        # res_B2[end-1:end,:] .= 0
        res_B = Hermitian(res_B1) + res_B2
        return res_B
        return Hermitian(res_B)
    end
end



function _fit_JuMP!(a::PSDModel{T}, 
                X::PSDDataVector{T}, 
                Y::Vector{T},
                W::Vector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                optimizer=nothing,
                mat_list = nothing,
                coef_list = nothing,
            ) where {T<:Number}
    return _least_squares_JuMP!(a, X, Y, W; λ_1=λ_1, λ_2=λ_2, trace=trace, optimizer=optimizer, mat_list=mat_list, coef_list=coef_list)
end


"""
Maximum likelihood implementation for JuMP solver.
"""
function _ML_JuMP!(a::PSDModel{T}, 
                samples::PSDDataVector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                optimizer=nothing,
                normalization=false,
                fixed_variables=nothing,
                mat_list = nothing,
                coef_list = nothing,
                set_start_value = true
            ) where {T<:Number}
    verbose_solver = trace ? true : false
    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            Hypatia.Optimizer,
        )
    else
        @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
    end

    model = JuMP.Model(optimizer)
    JuMP.set_string_names_on_creation(model, false)
    if verbose_solver
        JuMP.unset_silent(model)
    else
        JuMP.set_silent(model)
    end
    N = size(a.B, 1)
    JuMP.@variable(model, _B[1:N, 1:N], PSD)
    if isposdef(a.B) && set_start_value == true
        JuMP.set_start_value.(_B, a.B)
    end

    if mat_list !== nothing
        N_SoS_comb = [size(m[1], 2) for m in mat_list]
        D = []
        for i=1:length(mat_list)
            _D = JuMP.@variable(model, [1:N_SoS_comb[i], 1:N_SoS_comb[i]], PSD)
            push!(D, _D)
        end
        JuMP.@expression(model, SoS_comb, sum(
                                            sum(
                                                coef_list[i][k] * mat_list[i][k] * D[i] * mat_list[i][k]'
                                            for k=1:length(coef_list[i])) 
                                        for i=1:length(mat_list)))
    end

    B = if mat_list !== nothing
        _B + SoS_comb
    else
        _B
    end

    if fixed_variables !== nothing
        throw(@error "fixed variables not supported yet by JuMP interface!")
    end

    K = reduce(hcat, Φ.(Ref(a), samples))

    m = length(samples)

    JuMP.@expression(model, ex[i=1:m], K[:,i]' * B * K[:,i])

    
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [t; ex; (1/m)*ones(m)] in JuMP.MOI.RelativeEntropyCone(2*m+1))
    
    # JuMP.@variable(model, t[i=1:m])
    # JuMP.@constraint(model, [i=1:m], [t[i]; 1; ex[i]] in JuMP.MOI.ExponentialCone())


    JuMP.@expression(model, min_func, t + tr(B))

    
    if λ_1 > 0.0
        JuMP.add_to_expression!(min_func, λ_1 * nuclearnorm(B))
    end

    JuMP.@objective(model, Min, min_func)



    # JuMP.@objective(model, Min, min_func);

    # @show t2
    if normalization
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        JuMP.@constraint(model, tr(B) == 1)
    end

    JuMP.optimize!(model)
    res_B = if mat_list === nothing
        res_B = Hermitian(T.(JuMP.value(_B)))
        res_B
    else
        D = JuMP.value.(D)
        res_B1 = T.(JuMP.value(_B))
        res_B2 = sum(
                    sum(
                        coef_list[i][k] * mat_list[i][k] * T.(D[i]) * mat_list[i][k]'
                    for k=1:length(coef_list[i])) 
                for i=1:length(mat_list))
        res_B = res_B1 + res_B2
        res_B
    end
    set_coefficients!(a, Hermitian(res_B))
    _loss(Z) = -(1.0/length(Z)) * sum(log.(Z))

    finalize(model)
    model = nothing
    GC.gc()
    return _loss(a.(samples))
end


function _KL_JuMP!(a::PSDModel{T}, 
                X::PSDDataVector{T},
                Y::Vector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                optimizer=nothing,
                normalization=false,
                fixed_variables=nothing,
                mat_list = nothing,
                coef_list = nothing,
                set_start_value = true
            ) where {T<:Number}
    verbose_solver = trace ? true : false
    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            Hypatia.Optimizer,
        )
    else
        @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
    end

    model = JuMP.Model(optimizer)
    JuMP.set_string_names_on_creation(model, false)
    if verbose_solver
        JuMP.unset_silent(model)
    else
        JuMP.set_silent(model)
    end
    N = size(a.B, 1)
    JuMP.@variable(model, _B[1:N, 1:N], PSD)
    if set_start_value
        JuMP.set_start_value.(_B, a.B)
    end

    if mat_list !== nothing
        N_SoS_comb = [size(m[1], 2) for m in mat_list]
        D = []
        for i=1:length(mat_list)
            _D = JuMP.@variable(model, [1:N_SoS_comb[i], 1:N_SoS_comb[i]], PSD)
            push!(D, _D)
        end
        JuMP.@expression(model, SoS_comb, sum(
                                            sum(
                                                coef_list[i][k] * mat_list[i][k] * D[i] * mat_list[i][k]'
                                            for k=1:length(coef_list[i])) 
                                        for i=1:length(mat_list)))
    end

    if typeof(a) <: PSDOrthonormalSubModel
        @info "fix non marginal variables"
        JuMP.fix.(_B[map(~, a.M)], a.B[map(~, a.M)], force=true)
    end

    B = if typeof(a) <: PSDOrthonormalSubModel
        a.M .* _B
    elseif mat_list !== nothing
        _B + SoS_comb
    else
        _B
    end


    if fixed_variables !== nothing
        throw(@error "fixed variables not supported yet by JuMP interface!")
    end

    K = reduce(hcat, Φ.(Ref(a), X))

    m = length(X)

    JuMP.@expression(model, ex[i=1:m], K[:,i]' * B * K[:,i])

    
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [t; ex; Y] in JuMP.MOI.RelativeEntropyCone(2*m+1))
    
    # JuMP.@variable(model, t[i=1:m])
    # JuMP.@constraint(model, [i=1:m], [t[i]; 1; ex[i]] in JuMP.MOI.ExponentialCone())

    JuMP.@expression(model, min_func, t + tr(B))
    if λ_2 > 0.0
        JuMP.@expression(model, norm_B, sum(B[i,j]^2 for i=1:N, j=1:N))
        # JuMP.add_to_expression!(min_func, λ_2 * norm_B)
    end
    if λ_2 == 0.0
        JuMP.@objective(model, Min, min_func)
    else
        JuMP.@objective(model, Min, min_func + λ_2 * norm_B)
    end


    # JuMP.@objective(model, Min, min_func);

    # @show t2
    if normalization
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        JuMP.@constraint(model, tr(B) == 1)
    end


    JuMP.optimize!(model)
    res_B = if mat_list === nothing
        res_B = Hermitian(T.(JuMP.value(_B)))
        res_B
    else
        D = JuMP.value.(D)
        res_B1 = T.(JuMP.value(_B))
        res_B2 = sum(
                    sum(
                        coef_list[i][k] * mat_list[i][k] * T.(D[i]) * mat_list[i][k]'
                    for k=1:length(coef_list[i])) 
                for i=1:length(mat_list))
        res_B = res_B1 + res_B2
        res_B
    end
    set_coefficients!(a, Hermitian(res_B))
    _loss(Z) = (1.0/length(Z)) * sum(log.(Y./Z) .* Y .- Y) + tr(a.B)

    finalize(model)
    model = nothing
    GC.gc()
    return _loss(a.(X))
end


function _reversed_KL_JuMP!(a::PSDModel{T}, 
                X::PSDDataVector{T},
                Y::Vector{T};
                # λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                optimizer=nothing,
                normalization=false,
                fixed_variables=nothing,
                marg_constraints=nothing,
                marg_regularization=nothing,
                marg_data_regularization=nothing,
                λ_marg_reg=1.0,
                mat_list = nothing,
                coef_list = nothing,
                set_start_value = true
            ) where {T<:Number}
    verbose_solver = trace ? true : false
    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            Hypatia.Optimizer,
        )
    else
        @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
    end

    model = JuMP.Model(optimizer)
    JuMP.set_string_names_on_creation(model, false)
    if verbose_solver
        JuMP.unset_silent(model)
    else
        JuMP.set_silent(model)
    end
    N = size(a.B, 1)
    JuMP.@variable(model, _B[1:N, 1:N], PSD)
    if mat_list !== nothing
        N_SoS_comb = [size(m[1], 2) for m in mat_list]
        D = []
        for i=1:length(mat_list)
            _D = JuMP.@variable(model, [1:N_SoS_comb[i], 1:N_SoS_comb[i]], PSD)
            push!(D, _D)
        end
        JuMP.@expression(model, SoS_comb, sum(
                                            sum(
                                                coef_list[i][k] * mat_list[i][k] * D[i] * mat_list[i][k]'
                                            for k=1:length(coef_list[i])) 
                                        for i=1:length(mat_list)))
    end
    if set_start_value
        JuMP.set_start_value.(_B, a.B)
    end
    if typeof(a) <: PSDOrthonormalSubModel
        @info "fix non marginal variables"
        JuMP.fix.(_B[map(~, a.M)], a.B[map(~, a.M)], force=true)
    end

    B = if typeof(a) <: PSDOrthonormalSubModel
        a.M .* _B
    elseif mat_list !== nothing
        _B + SoS_comb
    else
        _B
    end


    if fixed_variables !== nothing
        @info "some variables are fixed"
        JuMP.fix.(B[fixed_variables], a.B[fixed_variables], force=true)
        # JuMP.@constraint(model, B[prob.fixed_variables] .== prob.initial[prob.fixed_variables])
    end

    K = reduce(hcat, Φ.(Ref(a), X))

    m = length(X)
    JuMP.@expression(model, ex[i=1:m], K[:,i]' * B * K[:,i])
    
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [t; Y; ex] in JuMP.MOI.RelativeEntropyCone(2*m+1))
    
    # JuMP.@variable(model, t[i=1:m])
    # JuMP.@constraint(model, [i=1:m], [t[i]; 1; ex[i]] in JuMP.MOI.ExponentialCone())

    if marg_constraints !== nothing
        for (marg_model, B_marg) in marg_constraints
            @info "fixing marginal"
            JuMP.@constraint(model, Hermitian(marg_model.P * (marg_model.M .* B) * marg_model.P') == B_marg)
        end
    end

    if marg_regularization !== nothing

        JuMP.@expression(model, marg_reg[i=1:length(marg_regularization)], 
                    (Hermitian(marg_regularization[i][1].P * (marg_regularization[i][1].M .* B) * 
                    marg_regularization[i][1].P') - marg_regularization[i][2]))

        JuMP.@variable(model, t_reg[i=1:length(marg_regularization)])
        for i=1:length(marg_regularization)
            JuMP.@constraint(model, [t_reg[i]; vec(marg_reg[i])] in JuMP.MOI.NormSpectralCone(size(marg_regularization[i][2])...))
        end
    end

    if marg_data_regularization !== nothing
        n_marg = length(marg_data_regularization)
        marg_model = [marg_data_regularization[i][1] for i=1:n_marg]
        X_i = [marg_data_regularization[i][2] for i=1:n_marg]
        K_marg = [reduce(hcat, Φ.(Ref(marg_model[i]), X_i[i])) for i=1:n_marg]

        m_marg = length(X_i[1])
        @assert all(length(X_i[i]) == m_marg for i=2:n_marg)
        JuMP.@expression(model, ex_marg[j=1:n_marg, i=1:m_marg], K_marg[j][:,i]' * (marg_model[j].M .* B) * K_marg[j][:,i])

        JuMP.@variable(model, t_marg[j=1:n_marg])
        for j=1:n_marg
            JuMP.@constraint(model, [t_marg[j]; ex_marg[j,:]; (1/m_marg)*ones(m_marg)] in JuMP.MOI.RelativeEntropyCone(2*m_marg+1))
        end
    end

    if marg_regularization !== nothing
        JuMP.@expression(model, min_func, t - tr(B) + λ_marg_reg * sum(t_reg))
    elseif marg_data_regularization !== nothing
        JuMP.@expression(model, min_func, t - tr(B) + 
                λ_marg_reg * (sum(t_marg) + sum(tr(marg_data_regularization[i][1].P * (marg_data_regularization[i][1].M .* B) * 
                marg_data_regularization[i][1].P') for i=1:length(marg_data_regularization))))
    else
        JuMP.@expression(model, min_func, t - tr(B))
    end

    if λ_2 > 0.0
        JuMP.@expression(model, norm_B, sum(B[i,j]^2 for i=1:N, j=1:N))
        # JuMP.add_to_expression!(min_func, λ_2 * norm_B)
    end
    if λ_2 == 0.0
        JuMP.@objective(model, Min, min_func)
    else
        JuMP.@objective(model, Min, min_func + λ_2 * norm_B)
    end


    # JuMP.@objective(model, Min, min_func);

    # @show t2
    if normalization
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        JuMP.@constraint(model, tr(B) == 1)
    end

    JuMP.optimize!(model)
    res_B = if mat_list === nothing
        res_B = Hermitian(T.(JuMP.value(_B)))
        res_B
    else
        D = JuMP.value.(D)
        res_B1 = T.(JuMP.value(_B))
        res_B2 = sum(
                    sum(
                        coef_list[i][k] * mat_list[i][k] * T.(D[i]) * mat_list[i][k]'
                    for k=1:length(coef_list[i])) 
                for i=1:length(mat_list))
        res_B = res_B1 + res_B2
        res_B
    end
    set_coefficients!(a, Hermitian(res_B))
    _loss(Z) = (1.0/length(Z)) * sum(log.(Y./Z) .* Y .- Y) + tr(a.B)

    finalize(model)
    model = nothing
    GC.gc()
    return _loss(a.(X))
end


function _OT_JuMP!(a::PSDModel{T}, 
                X::PSDDataVector{T},
                Y::Vector{T};
                trace=false,
                optimizer=nothing,
                normalization=false,
                fixed_variables=nothing,
                marg_constraints=nothing,
                marg_regularization=nothing,
                marg_data_regularization=nothing,
                α_marg=2.0,
                λ_marg_reg=1e5,
                mat_list = nothing,
                coef_list = nothing,
                model_for_marginals=nothing,
                set_start_value = true
            ) where {T<:Number}
    verbose_solver = trace ? true : false
    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            Hypatia.Optimizer,
        )
    else
        @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
    end

    d = length(X[1])
    _d = d ÷ 2

    model = JuMP.Model(optimizer)
    JuMP.set_string_names_on_creation(model, false)
    if verbose_solver
        JuMP.unset_silent(model)
    else
        JuMP.set_silent(model)
    end
    N = size(a.B, 1)
    JuMP.@variable(model, _B[1:N, 1:N], PSD)
    if set_start_value
        JuMP.set_start_value.(_B, a.B)
    end
    if mat_list !== nothing
        N_SoS_comb = [size(m[1], 2) for m in mat_list]
        # JuMP.@variable(model, D[1:length(mat_list), 1:N, 1:N])
        D = []
        for i=1:length(mat_list)
            _D = JuMP.@variable(model, [1:N_SoS_comb[i], 1:N_SoS_comb[i]], PSD)
            # JuMP.@constraint(model, D[i,:,:] in JuMP.PSDCone())
            push!(D, _D)
        end
        JuMP.@expression(model, SoS_comb, sum(
                                            sum(
                                                coef_list[i][k] * mat_list[i][k] * D[i] * mat_list[i][k]'
                                            for k=1:length(coef_list[i])) 
                                        for i=1:length(mat_list)))
    end

    B = if mat_list !== nothing
        _B + SoS_comb
    else
        _B
    end
    
    if fixed_variables !== nothing
        @info "some variables are fixed"
        JuMP.fix.(B[fixed_variables], a.B[fixed_variables], force=true)
        # JuMP.@constraint(model, B[prob.fixed_variables] .== prob.initial[prob.fixed_variables])
    end

    K = reduce(hcat, Φ.(Ref(a), X))

    m = length(X)
    JuMP.@expression(model, ex[i=1:m], K[:,i]' * B * K[:,i])
    
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [t; Y; ex] in JuMP.MOI.RelativeEntropyCone(2*m+1))

    if marg_constraints !== nothing
        for (marg_model, B_marg) in marg_constraints
            @info "fixing marginal"
            JuMP.@constraint(model, Hermitian(marg_model.P * (marg_model.M .* B) * marg_model.P') == B_marg)
        end
    end

    if (marg_data_regularization !== nothing || marg_regularization !== nothing)
      
        ## derive reduced matrx M
        quad_points, quad_weights = gausslegendre(50)
        quad_points = (quad_points .+ 1.0) * 0.5
        quad_weights = quad_weights * 0.5
        

        M = if model_for_marginals !== nothing
            (x) -> Φ(model_for_marginals, x) * Φ(model_for_marginals, x)'
        else
            (x) -> Φ(a, x) * Φ(a, x)'
        end
        
        marg_reg = if marg_regularization !== nothing
            marg_regularization
        else
            marg_data_regularization
        end

        n_marg = length(marg_reg)
        m_marg = length(marg_reg[1][2])
        @assert all(length(marg_reg[i][2]) == m_marg for i=1:n_marg)

        M_list = Matrix{Matrix{T}}(undef, n_marg, m_marg)

        for (j, marg_struct) in enumerate(marg_reg)
            e_j = marg_struct[1]
            e_mj = collect(1:d) .∉ Ref(e_j)
            e_mj = collect(1:d)[e_mj]
            _perm = [e_j; e_mj]
            @inline _assemble(x, y) = permute!([x; y], _perm)
            for (i, x) in enumerate(marg_struct[2])
                res = zeros(T, size(M(rand(d))))
                for k in Iterators.product([1:length(quad_points) for _ in 1:_d]...)
                    res += prod(quad_weights[[k...]]) * M(_assemble(x, quad_points[[k...]]))
                end
                M_list[j, i] = res
                # M_list[j, i] = sum(quad_weights .* map(q->M(_assemble(x, q)), quad_points))
            end
        end

        if marg_data_regularization !== nothing
            JuMP.@expression(model, ex_marg[j=1:n_marg, i=1:m_marg], dot(M_list[j, i], B))
            # JuMP.@expression(model, ex_marg[j=1:n_marg, i=1:m_marg], tr(M_list[j, i] * B))
            JuMP.@variable(model, t_marg[j=1:n_marg])
            for j=1:n_marg
                JuMP.@constraint(model, [t_marg[j]; ex_marg[j,:]; (1/m_marg)*ones(m_marg)] in JuMP.MOI.RelativeEntropyCone(2*m_marg+1))
            end
        else
            JuMP.@expression(model, ex_marg[j=1:n_marg, i=1:m_marg], dot(M_list[j, i], B))
            JuMP.@variable(model, t_marg[j=1:n_marg])
            JuMP.@variable(model, r[j=1:n_marg, i=1:m_marg])
            @assert α_marg != 1 || α_marg != 0
            if 0 < α_marg < 1
                JuMP.@constraint(model, [j=1:n_marg, i=1:m_marg], [marg_reg[j][3][i]; ex_marg[j, i]; r[j, i]] in JuMP.MOI.PowerCone(α_marg))
            elseif α_marg > 1
                JuMP.@constraint(model, [j=1:n_marg, i=1:m_marg], [r[j, i]; ex_marg[j, i]; marg_reg[j][3][i]] in JuMP.MOI.PowerCone(1/α_marg))
            else
                JuMP.@constraint(model, [j=1:n_marg, i=1:m_marg], [r[j, i]; one(T); marg_reg[j][3][i]^(α_marg/(1-α_marg)) * ex_marg[j, i]] in JuMP.MOI.PowerCone(1/(1-α_marg)))
            end
            JuMP.@constraint(model, [j=1:n_marg], t_marg[j] == (1/m) * sum(r[j,:]))
        end

    end

    if marg_regularization !== nothing
        JuMP.@expression(model, min_func, t - tr(B) + 
                λ_marg_reg * ((one(T)/(α_marg*(α_marg-one(T)))) * sum(t_marg) + (one(T)/α_marg) * 2 * tr(B)))
    elseif (marg_data_regularization) !== nothing
        JuMP.@expression(model, min_func, t - tr(B) + 
                λ_marg_reg * (sum(t_marg) + 2 * tr(B)))
    else
        JuMP.@expression(model, min_func, t - tr(B))
    end

    JuMP.@objective(model, Min, min_func)

    # @show t2
    if normalization
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        JuMP.@constraint(model, tr(B) == 1)
    end

    JuMP.optimize!(model)
    res_B = if mat_list === nothing
        res_B = Hermitian(T.(JuMP.value(_B)))
        res_B
    else
        D = JuMP.value.(D)
        res_B1 = T.(JuMP.value(_B))
        res_B2 = sum(
                    sum(
                        coef_list[i][k] * mat_list[i][k] * T.(D[i]) * mat_list[i][k]'
                    for k=1:length(coef_list[i])) 
                for i=1:length(mat_list))
        res_B = res_B1 + res_B2
        res_B
    end
    set_coefficients!(a, Hermitian(res_B))
    _loss(Z) = (1.0/length(Z)) * sum(log.(Y./Z) .* Y .- Y) + tr(a.B)

    finalize(model)
    model = nothing
    GC.gc()
    return _loss(a.(X))
end



function _α_divergence_JuMP!(a::PSDModel{T},
                α::T,
                X::PSDDataVector{T},
                Y::Vector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                optimizer=nothing,
                normalization=false,
                fixed_variables=nothing,
                marg_constraints=nothing,
                mat_list = nothing,
                coef_list = nothing,
                set_start_value = true
            ) where {T<:Number}
    verbose_solver = trace ? true : false
    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            Hypatia.Optimizer,
        )
    else
        @info "optimizer is given, optimizer parameters are ignored. If you want to set them, use MOI.OptimizerWithAttributes."
    end

    model = JuMP.Model(optimizer)
    JuMP.set_string_names_on_creation(model, false)
    if verbose_solver
        JuMP.unset_silent(model)
    else
        JuMP.set_silent(model)
    end
    N = size(a.B, 1)
    JuMP.@variable(model, _B[1:N, 1:N], PSD)

    if set_start_value
        JuMP.set_start_value.(_B, a.B)
    end
    if mat_list !== nothing
        N_SoS_comb = [size(m[1], 2) for m in mat_list]
        # JuMP.@variable(model, D[1:length(mat_list), 1:N, 1:N])
        D = []
        for i=1:length(mat_list)
            _D = JuMP.@variable(model, [1:N_SoS_comb[i], 1:N_SoS_comb[i]], PSD)
            # JuMP.@constraint(model, D[i,:,:] in JuMP.PSDCone())
            push!(D, _D)
        end
        JuMP.@expression(model, SoS_comb, sum(
                                            sum(
                                                coef_list[i][k] * mat_list[i][k] * D[i] * mat_list[i][k]'
                                            for k=1:length(coef_list[i])) 
                                        for i=1:length(mat_list)))
    end

    B = if mat_list !== nothing
        _B + SoS_comb
    else
        _B
    end
    if fixed_variables !== nothing
        throw(@error "fixed variables not supported yet by JuMP interface!")
    end

    K = reduce(hcat, Φ.(Ref(a), X))

    m = length(X)
    JuMP.@expression(model, ex[i=1:m], K[:,i]' * B * K[:,i])

    JuMP.@variable(model, t)
    JuMP.@variable(model, r[1:m])
    if 0 < α < 1
        JuMP.@constraint(model, [i=1:m], [Y[i]; ex[i]; r[i]] in JuMP.MOI.PowerCone(α))
    elseif α > 1
        JuMP.@constraint(model, [i=1:m], [r[i]; ex[i]; Y[i]] in JuMP.MOI.PowerCone(1/α))
    else
        JuMP.@constraint(model, [i=1:m], [r[i]; one(T); Y[i]^(α/(1-α)) * ex[i]] in JuMP.MOI.PowerCone(1/(1-α)))
    end
    JuMP.@constraint(model, t == (1/m) * sum(r))

    JuMP.@expression(model, min_func, (one(T)/(α*(α-one(T)))) * t + (1/α)* tr(B))
    ## use discrete approximation of the integral
    # JuMP.@expression(model, min_func, (one(T)/(α*(α-one(T)))) * t + (1/α)* sum(ex[i] for i=1:m))
    if λ_2 > 0.0
        JuMP.@expression(model, norm_B, sum(B[i,j]^2 for i=1:N, j=1:N))
        # JuMP.add_to_expression!(min_func, λ_2 * norm_B)
    end
    if λ_2 == 0.0
        JuMP.@objective(model, Min, min_func)
    else
        JuMP.@objective(model, Min, min_func + λ_2 * norm_B)
    end


    # JuMP.@objective(model, Min, min_func);

    # @show t2
    if normalization
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        JuMP.@constraint(model, tr(B) == 1)
    end

    if marg_constraints !== nothing
        for (marg_model, B_marg) in marg_constraints
            @info "fixing marginal"
            JuMP.@constraint(model, Hermitian(marg_model.P * (marg_model.M .* B) * marg_model.P') == B_marg)
        end
    end

    JuMP.optimize!(model)
    res_B = if mat_list === nothing
        res_B = Hermitian(T.(JuMP.value(_B)))
        res_B
    else
        D = JuMP.value.(D)
        res_B1 = T.(JuMP.value(_B))
        res_B2 = sum(
                    sum(
                        coef_list[i][k] * mat_list[i][k] * T.(D[i]) * mat_list[i][k]'
                    for k=1:length(coef_list[i])) 
                for i=1:length(mat_list))
        res_B = res_B1 + res_B2
        res_B
    end
    set_coefficients!(a, Hermitian(res_B))
    _loss(Z) = (1.0/length(Z)) * (one(T)/(α*(α-one(T)))) * sum(Z.^(one(T)-α) .* Y.^(α) .- α * Y) + (one(T)/α) * tr(a.B)

    finalize(model)
    model = nothing
    GC.gc()
    return _loss(a.(X))
end
