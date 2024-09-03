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
            maxit::Int=5000,
            marg_constraints = nothing,
        ) where {T<:Number}
        if optimizer === nothing
            optimizer = con.MOI.OptimizerWithAttributes(
                SCS.Optimizer,
                "max_iters" => maxit,
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

function _interpolate_JuMP!(a::PSDModel{T},
                    X::PSDDataVector{T},
                    Y::Vector{T};
                    λ_1 = 0.0,
                    λ_2 = 0.0,
                    trace=false,
                    optimizer=nothing,
                    maxit=5000,mat_list = nothing,
                    coef_list = nothing) where {T<:Number}
    verbose_solver = trace ? true : false

    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            SCS.Optimizer,
            "max_iters" => maxit,
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
    # print("Condition number of M is ", cond(M),"\n")
    # trace && print("Calculate non PSD estimate of B...")
    # t = time()
    # res_B_vec = M \ Y
    # dt = time() - t
    # trace && print("done! - $(dt) \n")
    # res_B_vec = reshape(res_B_vec, length(res_B_vec))


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

    JuMP.set_start_value.(B, a.B)
    B_red = Hermitian_to_low_vec(B)

    JuMP.@variable(model, t)
    size_cone = length(Y)

    if mat_list !== nothing
        SoS_red = Hermitian_to_low_vec(SoS_comb)
        JuMP.@constraint(model, [t; M * (B_red + SoS_red) - Y] in JuMP.MOI.SecondOrderCone(size_cone+1))
    else
        JuMP.@constraint(model, [t; M * B_red - Y] in JuMP.MOI.SecondOrderCone(size_cone+1))
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

function _least_squares_JuMP!(a::PSDModel{T},
                    X::PSDDataVector{T},
                    Y::Vector{T};
                    λ_1 = 0.0,
                    λ_2 = 0.0,
                    trace=false,
                    optimizer=nothing,
                    maxit=5000,) where {T<:Number}
    verbose_solver = trace ? true : false

    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            SCS.Optimizer,
            "max_iters" => maxit,
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
    M2 = M' * M
    Y2 = M' * Y
    # print("Condition number of M is ", cond(M),"\n")
    # trace && print("Calculate non PSD estimate of B...")
    # t = time()
    # res_B_vec = M \ Y
    # dt = time() - t
    # trace && print("done! - $(dt) \n")
    # res_B_vec = reshape(res_B_vec, length(res_B_vec))


    N = size(a.B, 1)
    JuMP.@variable(model, B[1:N, 1:N], PSD)

    JuMP.set_start_value.(B, a.B)
    B_red = Hermitian_to_low_vec(B)
    JuMP.@variable(model, t)
    size_cone = length(Y2)
 
    JuMP.@constraint(model, [t; M2 * B_red - Y2] in JuMP.MOI.SecondOrderCone(size_cone+1))
    JuMP.@objective(model, Min, t)
    JuMP.optimize!(model)

    res_B = Hermitian(T.(JuMP.value(B)))
    # e_vals, e_vecs = eigen(res_B)
    # e_vals[e_vals .< 0.0] .= 0.0
    # res_B = e_vecs * Diagonal(e_vals) * e_vecs'
    set_coefficients!(a, Hermitian(res_B))
    _loss(Z) = (1.0/length(Z)) * sum((Z .- Y).^2)

    finalize(model)
    model = nothing
    GC.gc()
    return _loss(a.(X))
end

function _closest_PSD_JuMP!(A::Hermitian{T}; optimizer=nothing, maxit=10000,
            mat_list = nothing, coef_list = nothing,
    ) where {T<:Number}

    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            SCS.Optimizer,
            "max_iters" => maxit,
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
                        coef_list[i][k] * mat_list[i][k] * T.(D[i,:,:]) * mat_list[i][k]'
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
                weights::Vector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
                optim_threading=false, # only for problem setup
                trace=false,
                optimizer=nothing,
                maxit=5000,
                normalization=false,
                fixed_variables=nothing,
            ) where {T<:Number}
    verbose_solver = trace ? true : false
    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            SCS.Optimizer,
            "max_iters" => maxit,
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
    JuMP.@variable(model, B[1:N, 1:N], PSD)

    JuMP.set_start_value.(B, a.B)
    if fixed_variables !== nothing
        throw(@error "fixed variables not supported yet by JuMP interface!")
    end

    K = reduce(hcat, Φ.(Ref(a), X))

    _K_tmp = similar(K)
    for i=1:N
        _K_tmp[i, :] = K[i,:] .* weights
    end
    kv1_f(j, l) = begin
        ret = zero(T)
        @inbounds @fastmath @simd for i=1:length(Y)
            ret += K[j,i] * K[j,i] * K[l,i] * _K_tmp[l, i]
        end
        return ret
    end
    kv2_f(j::Int, l::Int, j2::Int, l2::Int) = begin
        ret = zero(T)
        @inbounds @fastmath @simd for i=1:length(Y)
            ret += K[j,i] * K[l,i] * K[j2,i] * _K_tmp[l2,i]
        end
        return ret
    end
    kv_f(j, l) = begin
        ret = zero(T)
        @inbounds @fastmath @simd for i=1:length(Y)
            ret += K[j,i] * _K_tmp[l,i] * Y[i]
        end
        return ret
    end

    JuMP.@expression(
        model, 
        ex, 
        # sum( j!=l ? 2.0 * _K_vec1[j, l] * B[j, l]^2 : 0.0 for j=1:N, l=1:N)
        + sum(kv1_f(j, j) * B[j, j]^2 for j=1:N)
        - 2.0 * sum( 
            kv_f(j, l) * B[j, l] for j=1:N, l=1:N
        ) 
    )
    # use the lock also when not threading, performance overhead should be small
    if optim_threading
        lk = Threads.ReentrantLock()
        Threads.@threads for j=1:N
            for l=1:(j-1)
                coeff_vec = Vector{T}(undef, N)
                for j2=1:N
                    coeff_vec[j2] = 4.0 * kv2_f(j, l, j2, j2)
                end
                coeff1 = 4.0 * kv1_f(j, l)
                coeff2 = 2.0 * kv2_f(j, j, l, l)
                lock(lk) do
                    JuMP.add_to_expression!(ex, sum(coeff_vec[i] * B[j, l] * B[i,i] for i=1:N))
                    JuMP.add_to_expression!(ex, coeff1 * B[j, l]^2)
                    JuMP.add_to_expression!(ex, coeff2 * B[j, j] * B[l, l])
                end
                for l2=1:(l-1)
                    coeff_vec[l2] = 8.0 * kv2_f(j, l, j, l2)
                end
                if (l-1)>0
                    lock(lk) do
                        JuMP.add_to_expression!(ex, sum(coeff_vec[i] * B[j, l] * B[j, i] for i=1:(l-1)))
                    end
                end
                for j2=1:(j-1), l2=1:(j2-1)
                    coeff = 8.0 * kv2_f(j, l, j2, l2)
                    lock(lk) do
                        JuMP.add_to_expression!(ex, coeff * B[j, l] * B[j2, l2])
                    end
                end
            end
        end
    else
        for j=1:N
            for l=1:(j-1)
                coeff = 4.0 * kv1_f(j, l)
                JuMP.add_to_expression!(ex, coeff * B[j, l]^2)
            end

            for j2=1:(j-1)
                coeff = 2.0 * kv2_f(j, j, j2, j2)
                JuMP.add_to_expression!(ex, coeff * B[j, j] * B[j2, j2])
            end

            for l=1:(j-1)
                for j2=1:N
                    coeff = 4.0 * kv2_f(j, l, j2, j2)
                    JuMP.add_to_expression!(ex, coeff * B[j, l] * B[j2, j2])
                end
                for l2=1:(l-1)
                    coeff = 8.0 * kv2_f(j, l, j, l2)
                    JuMP.add_to_expression!(ex, coeff * B[j, l] * B[j, l2])
                end
                for j2=1:(j-1), l2=1:(j2-1)
                    coeff = 8.0 * kv2_f(j, l, j2, l2)
                    JuMP.add_to_expression!(ex, coeff * B[j, l] * B[j2, l2])
                end
            end
        end
    end

    if λ_1 > 0.0
        JuMP.add_to_expression!(ex, λ_1 * nuclearnorm(B))
    end
    if λ_2 > 0.0
        JuMP.@expression(model, norm_B, sum(B[i,j]^2 for i=1:N, j=1:N))
        # JuMP.add_to_expression!(min_func, λ_2 * norm_B)
    end
    if λ_2 == 0.0
        JuMP.@objective(model, Min, ex)
    else
        JuMP.@objective(model, Min, ex + λ_2 * norm_B)
    end

    # @show t2
    if normalization
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        JuMP.@constraint(model, tr(B) == 1)
    end

    JuMP.optimize!(model)
    res_B = Hermitian(T.(JuMP.value(B)))
    e_vals, e_vecs = eigen(res_B)
    e_vals[e_vals .< 0.0] .= 0.0
    res_B = e_vecs * Diagonal(e_vals) * e_vecs'
    set_coefficients!(a, Hermitian(res_B))
    _loss(Z) = (1.0/length(Z)) * sum((Z .- Y).^2 .* weights)

    finalize(model)
    model = nothing
    GC.gc()
    return _loss(a.(X))
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
                maxit=5000,
                normalization=false,
                fixed_variables=nothing,
                mat_list = nothing,
                coef_list = nothing
            ) where {T<:Number}
    verbose_solver = trace ? true : false
    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            SCS.Optimizer,
            "max_iters" => maxit,
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
    if isposdef(a.B)
        JuMP.set_start_value.(_B, a.B)
    end

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
                        coef_list[i][k] * mat_list[i][k] * T.(D[i,:,:]) * mat_list[i][k]'
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
                maxit=5000,
                normalization=false,
                fixed_variables=nothing,
                mat_list = nothing,
                coef_list = nothing,
            ) where {T<:Number}
    verbose_solver = trace ? true : false
    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            SCS.Optimizer,
            "max_iters" => maxit,
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
    JuMP.set_start_value.(_B, a.B)

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
                        coef_list[i][k] * mat_list[i][k] * T.(D[i,:,:]) * mat_list[i][k]'
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
                maxit=5000,
                normalization=false,
                fixed_variables=nothing,
                marg_constraints=nothing,
                marg_regularization=nothing,
                marg_data_regularization=nothing,
                λ_marg_reg=1.0,
                mat_list = nothing,
                coef_list = nothing,
            ) where {T<:Number}
    verbose_solver = trace ? true : false
    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            SCS.Optimizer,
            "max_iters" => maxit,
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
    JuMP.set_start_value.(_B, a.B)
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
                        coef_list[i][k] * mat_list[i][k] * T.(D[i,:,:]) * mat_list[i][k]'
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
                # λ_1 = 0.0,
                λ_2 = 0.0,
                trace=false,
                optimizer=nothing,
                maxit=5000,
                normalization=false,
                fixed_variables=nothing,
                marg_constraints=nothing,
                marg_regularization=nothing,
                marg_data_regularization=nothing,
                λ_marg_reg=1.0,
                mat_list = nothing,
                coef_list = nothing,
                model_for_marginals=nothing,
            ) where {T<:Number}
    verbose_solver = trace ? true : false
    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            SCS.Optimizer,
            "max_iters" => maxit,
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
    JuMP.set_start_value.(_B, a.B)
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
        m_marg = length(marg_data_regularization[1][2])
        @assert all(length(marg_data_regularization[i][2]) == m_marg for i=1:n_marg)
        
        ## derive reduced matrx M
        quad_points, quad_weights = gausslegendre(30)
        quad_points = (quad_points .+ 1.0) * 0.5
        quad_weights = quad_weights * 0.5
        M = if model_for_marginals !== nothing
            (x) -> Φ(model_for_marginals, x) * Φ(model_for_marginals, x)'
        else
            (x) -> Φ(a, x) * Φ(a, x)'
        end
        M_list = Matrix{Matrix{T}}(undef, n_marg, m_marg)
        for (j, marg_struct) in enumerate(marg_data_regularization)
            e_j = marg_struct[1]
            e_mj = (ones(2) - e_j)
            for (i, x) in enumerate(marg_struct[2])
                fix_x = e_j * x[1]
                M_list[j, i] = sum(quad_weights .* map(q->M(fix_x + e_mj * q), quad_points))
            end
        end

        JuMP.@expression(model, ex_marg[j=1:n_marg, i=1:m_marg], dot(M_list[j, i], B))
        # JuMP.@expression(model, ex_marg[j=1:n_marg, i=1:m_marg], tr(M_list[j, i] * B))
        JuMP.@variable(model, t_marg[j=1:n_marg])
        for j=1:n_marg
            JuMP.@constraint(model, [t_marg[j]; ex_marg[j,:]; (1/m_marg)*ones(m_marg)] in JuMP.MOI.RelativeEntropyCone(2*m_marg+1))
        end


        # n_marg = length(marg_data_regularization)
        # marg_model = [marg_data_regularization[i][1] for i=1:n_marg]
        # X_i = [marg_data_regularization[i][2] for i=1:n_marg]
        # K_marg = [reduce(hcat, Φ.(Ref(marg_model[i]), X_i[i])) for i=1:n_marg]

        # m_marg = length(X_i[1])
        # @assert all(length(X_i[i]) == m_marg for i=2:n_marg)
        # JuMP.@expression(model, ex_marg[j=1:n_marg, i=1:m_marg], K_marg[j][:,i]' * (marg_model[j].M .* B) * K_marg[j][:,i])

        # JuMP.@variable(model, t_marg[j=1:n_marg])
        # for j=1:n_marg
        #     JuMP.@constraint(model, [t_marg[j]; ex_marg[j,:]; (1/m_marg)*ones(m_marg)] in JuMP.MOI.RelativeEntropyCone(2*m_marg+1))
        # end
    end

    if marg_regularization !== nothing
        JuMP.@expression(model, min_func, t - tr(B) + λ_marg_reg * sum(t_reg))
    elseif marg_data_regularization !== nothing
        JuMP.@expression(model, min_func, t - tr(B) + 
                λ_marg_reg * (sum(t_marg) + 2*tr(B)))
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
                        coef_list[i][k] * mat_list[i][k] * T.(D[i,:,:]) * mat_list[i][k]'
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
                maxit=5000,
                normalization=false,
                fixed_variables=nothing,
                marg_constraints=nothing,
                mat_list = nothing,
                coef_list = nothing,
            ) where {T<:Number}
    verbose_solver = trace ? true : false
    if optimizer===nothing
        optimizer = con.MOI.OptimizerWithAttributes(
            SCS.Optimizer,
            "max_iters" => maxit,
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

    JuMP.set_start_value.(_B, a.B)
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
                        coef_list[i][k] * mat_list[i][k] * T.(D[i,:,:]) * mat_list[i][k]'
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
