import JuMP

struct JuMPOptProp{T} <: OptProp{T}
    initial::AbstractMatrix{T}
    loss::Function
    normalization::Bool           # if tr(X) = 1
    optimizer
    fixed_variables
    trace::Bool
    function JuMPOptProp(
            initial::AbstractMatrix{T}, 
            loss::Function;
            trace=false,
            optimizer=nothing,
            fixed_variables=nothing,
            normalization=false,
            maxit::Int=5000,
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
                trace
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
        throw(@error "fixed variables not supported yet by JuMP interface!")
    end
    JuMP.@objective(model, Min, prob.loss(B))

    if prob.normalization
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        JuMP.@constraint(model, tr(B) == 1)
    end

    JuMP.optimize!(model)
    return Hermitian(T.(JuMP.value(B)))
end


function _fit_JuMP!(a::PSDModel{T}, 
                X::PSDDataVector{T}, 
                Y::Vector{T},
                weights::Vector{T};
                λ_1 = 0.0,
                λ_2 = 0.0,
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
    for j=1:N
        for l=1:(j-1)
            coeff = 4.0 * kv1_f(j, l)
            JuMP.add_to_expression!(ex, coeff * B[j, l]^2)
        end
    end
    for j=1:N
        for j2=1:(j-1)
            coeff = 2.0 * kv2_f(j, j, j2, j2)
            JuMP.add_to_expression!(ex, coeff * B[j, j] * B[j2, j2])
        end
    end
    for j=1:N, l=1:(j-1)
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


    if λ_1 > 0.0
        JuMP.add_to_expression!(ex, λ_1 * nuclearnorm(B))
    end
    if λ_2 > 0.0
        JuMP.add_to_expression!(ex, λ_2 * opnorm(B, 2)^2)
    end

    JuMP.@objective(model, Min, ex);

    # @show t2
    if normalization
        # IMPORTANT: only valid for tensorized polynomial maps.
        @info "s.t. tr(B) = 1 used, only valid for tensorized polynomial maps as normalization constraint."
        JuMP.@constraint(model, tr(B) == 1)
    end

    JuMP.optimize!(model)
    set_coefficients!(a, Hermitian(T.(JuMP.value(B))))
    _loss(Z) = (1.0/N) * sum((Z .- Y).^2 .* weights)
    return 
end
