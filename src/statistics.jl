module Statistics

using ..SequentialMeasureTransport
import ..SequentialMeasureTransport as SMT
using ..SequentialMeasureTransport: PSDDataVector
using ..SequentialMeasureTransport: PSDModelOrthonormal, PSDModelPolynomial
using ..SequentialMeasureTransport: CondSampler
using ..SequentialMeasureTransport: domain_interval_left, domain_interval_right
using ..SequentialMeasureTransport: greedy_IRLS
using ..SequentialMeasureTransport: _ML_JuMP!, _ML_Manopt!
using ..SequentialMeasureTransport: _KL_JuMP!, _reversed_KL_JuMP!, _KL_Manopt!
using ..SequentialMeasureTransport: _α_divergence_JuMP!, _α_divergence_Manopt!
using LinearAlgebra
using FastGaussQuadrature: gausslegendre
using Distributions: pdf

export ML_fit!, Chi2_fit!, TV_fit!, KL_fit!, Hellinger_fit!, α_divergence_fit!
export reversed_KL_fit!

"""
    ML_fit!(model, samples; kwargs...)

Maximum likelihood fit of a PSD model to the samples.
"""
function ML_fit!(model::PSDModel{T}, 
        samples::PSDDataVector{T};
        SDP_library=:JuMP,
        kwargs...) where {T<:Number}
    if SDP_library == :JuMP
        if typeof(model) <: PSDModelPolynomial
            D, C = SMT.get_semialgebraic_domain_constraints(model)
            return _ML_JuMP!(model, samples; normalization=true, mat_list=D, coef_list=C, kwargs...)
        end
        return _ML_JuMP!(model, samples; normalization=true, kwargs...)
    elseif SDP_library == :Manopt
        return _ML_Manopt!(model, samples; kwargs...)
    end

    loss_KL(Z) = -(1/length(Z)) * sum(log.(Z))
    minimize!(model, loss_KL, samples; 
            normalization_constraint=true,
            kwargs...)
end

"""
    Chi2_fit!(model, samples; kwargs...)

fit of unnormalized distribution. The loss function is defined by
Z_y/Z_f^2 * ∫ (f(x) - y(x))^2/y(x) dx
"""
function Chi2_fit!(model::PSDModel{T}, 
    X::PSDDataVector{T},
    Y::AbstractVector{T};
    ϵ=1e-5, data_normalization=true,
    IRLS=true,
    kwargs...) where {T<:Number}
    if data_normalization
        Y = Y ./ sum(Y)
    end

    if IRLS
        # Chi2 defined by Z_y / Z_f^2 ∫ (f(x) - y(x))^2/y(x) dx
        # => IRLS with weights 1/(y(x) + ϵ), ϵ for numerical reasons

        # Reweighting of the IRLS algorithm
        reweight(Z, B) = tr(B) ./ (abs.(Z) .+ ϵ)
    
        IRLS!(model, X, Y, reweight; reweight_include_B=true, kwargs...)

    else
        loss(Z, I_Z) = I_Z^2 - 2.0 * I_Z + (1/length(Z)) * sum(Y.^2 ./ (Z .+ ϵ))
        minimize!(model, loss, X; L_includes_normalization=true, kwargs...)
    end
end

function Hellinger_fit!(model::PSDModel{T}, 
        X::PSDDataVector{T},
        Y::AbstractVector{T};
        data_normalization=true,
        SDP_library=:JuMP,
        kwargs...) where {T<:Number}
    if data_normalization
        Y = Y ./ sum(Y)
    end
    if SDP_library == :JuMP
        return α_divergence_fit!(model, T(0.5), X, Y; SDP_library=SDP_library,kwargs...)
    end
    loss_Hellinger(Z) = (1/length(Z)) * 0.5 * sum(Z.+Y.-2.0*sqrt.(Z.*Y))
    minimize!(model, loss_Hellinger, X; kwargs...)
end

function TV_fit!(model::PSDModel{T},
    X::PSDDataVector{T},
    Y::AbstractVector{T};
    ϵ=1e-5,
    data_normalization=true,
    kwargs...) where {T<:Number}
    if data_normalization
        Y = Y ./ sum(Y)
    end
    
    reweight(Z) = 1 ./ (abs.(Z .- Y) .+ ϵ)

    IRLS!(model, X, Y, reweight; kwargs...)
end

function α_divergence_fit!(model::PSDModel{T},
        α::T,
        X::PSDDataVector{T},
        Y::AbstractVector{T};
        data_normalization=true,
        SDP_library=:JuMP,
        kwargs...
    ) where {T<:Number}
    if data_normalization
        Y = Y ./ sum(Y)
    end
    
    if SDP_library == :JuMP
        if typeof(model) <: PSDModelPolynomial
            D, C = SMT.get_semialgebraic_domain_constraints(model)
            return _α_divergence_JuMP!(model, α, X, Y; mat_list=D, coef_list=C, kwargs...)
        end
        return _α_divergence_JuMP!(model, α, X, Y; kwargs...)
    elseif SDP_library == :Manopt
        return _α_divergence_Manopt!(model, α, X, Y; kwargs...)
    else
        throw(ArgumentError("Only JuMP and Manopt is supported for now."))
    end
end


"""
KL-divergence extended to positive measures, defined by the alpha-divergence.
"""
function KL_fit!(model::PSDModel{T},
        X::PSDDataVector{T},
        Y::AbstractVector{T};
        normalization_constraint=false,
        SDP_library=:JuMP,
        data_normalization=false,
        kwargs...) where {T<:Number}
    # Helps some solvers, such as SCS, not needed with Hypatia
    if data_normalization
        Y = Y ./ sum(Y)
    end
    if SDP_library == :JuMP
        if typeof(model) <: PSDModelPolynomial
            D, C = SMT.get_semialgebraic_domain_constraints(model)
            return _KL_JuMP!(model, X, Y; 
                        normalization=normalization_constraint;
                        mat_list=D, coef_list=C, kwargs...)
        end
        return _KL_JuMP!(model, X, Y; 
                        normalization=normalization_constraint, 
                        kwargs...)
    elseif SDP_library == :Manopt
        return _KL_Manopt!(model, X, Y; 
                        normalization=normalization_constraint, 
                        kwargs...)
    elseif SDP_library == :Convex
        # KL by α div: \int log(f/g) df - \int df + \int dg
        # we deal with \int dg = tr(B), hence setting λ_1 = 1.0
        loss(Z) = (1/length(Z)) * sum(-log.(Z) .* Y)
        if normalization_constraint
            minimize!(model, loss, X; 
                normalization_constraint=normalization_constraint,
                kwargs...)
        else
            minimize!(model, loss, X; 
                normalization_constraint=false,
                λ_1 = 1.0,
                kwargs...)
        end
    else
        throw(ArgumentError("Only JuMP, Manopt and Convex is supported for now."))
    end
end

"""
KL-divergence extended to positive measures, defined by the alpha-divergence.
"""
function reversed_KL_fit!(model::PSDModel{T},
        X::PSDDataVector{T},
        Y::AbstractVector{T};
        normalization_constraint=false,
        SDP_library=:JuMP,
        data_normalization=true,
        kwargs...) where {T<:Number}
    # Helps some solvers, such as SCS, not needed with Hypatia
    if data_normalization
        Y = Y ./ sum(Y)
    end
    if SDP_library == :JuMP
        if typeof(model) <: PSDModelPolynomial
            D, C = SMT.get_semialgebraic_domain_constraints(model)
            return _reversed_KL_JuMP!(model, X, Y; 
                        normalization=normalization_constraint;
                        mat_list=D, coef_list=C, kwargs...)
        end
        return _reversed_KL_JuMP!(model, X, Y; 
                        normalization=normalization_constraint, 
                        kwargs...)
    end
    if normalization_constraint == true
        @info "By using normalization during minimization, the KL-divergence for probabilities is used."
    end
    # KL by α div: \int log(f/g) df - \int df + \int dg
    # we deal with \int dg = tr(B), hence setting λ_1 = 1.0
    loss(Z) = (1/length(Z)) * sum((log.(Z) - log.(Y)) .* Z)
    if normalization_constraint
        minimize!(model, loss, X; 
            normalization_constraint=normalization_constraint,
            kwargs...)
    else
        minimize!(model, loss, X; 
            normalization_constraint=false,
            λ_1 = 1.0,
            kwargs...)
    end
end

function conditional_expectation(model::PSDModelOrthonormal{d, T}, 
    dim::Int) where {d, T<:Number}
    if d==1
        return @error("model only has one variable, the conditional expectation does not exist. Use expectation instead.")
    end
    weight_func = x -> x
    return marginalize(model, dim, weight_func)
end

function expectation(model::PSDModelOrthonormal{d, T}, 
    dim::Int) where {d, T<:Number}
    weight_func = x -> x
    mod = marginalize(model, dim, weight_func)
    if d==1
        return mod
    else
        return marginalize(mod)
    end
end

function expectation(model::PSDModelOrthonormal{d, T}) where {d, T<:Number}
    expectation_values = zeros(T, d)
    for i in 1:d
        expectation_values[i] = expectation(model, i)
    end
    return expectation_values
end

function conditional_covariance(model::PSDModelOrthonormal{d, T},
    dim1::Int, dim2::Int) where {d, T<:Number}
    @assert 1 ≤ dim1 ≤ d
    @assert 1 ≤ dim2 ≤ d
    if (dim1 == dim2 && d==1) || (dim1 != dim2 && d==2)
        return @error "no dimension left, covariance is not conditional, use covariance function instead"
    end
    E_1 = conditional_expectation(model, dim1)
    E_2 = conditional_expectation(model, dim2)
    if dim1 == dim2
        # nothing to do
    elseif dim1 > dim2
        E_1 = marginalize(E_1, dim2)
        E_2 = marginalize(E_2, dim1-1)
    else
        E_1 = marginalize(E_1, dim2-1)
        E_2 = marginalize(E_2, dim1)
    end

    E_12 = if dim1 == dim2
        weight_func = x -> x^2
        marginalize(model, dim1, weight_func)
    elseif dim1 > dim2
        weight_func = x -> x
        mod = marginalize(model, dim1, weight_func)
        marginalize(mod, dim2, weight_func)
    else
        weight_func = x -> x
        mod = marginalize(model, dim2, weight_func)
        marginalize(mod, dim1, weight_func)
    end
    covariance_func = let E_1=E_1, E_2=E_2, E_12=E_12
        x -> E_12(x) - E_1(x) * E_2(x)
    end
    return covariance_func
end

function covariance(model::PSDModelOrthonormal{d, T},
    dim1::Int, dim2::Int) where {d, T<:Number}
    @assert 1 ≤ dim1 ≤ d
    @assert 1 ≤ dim2 ≤ d
    E_1 = expectation(model, dim1)
    E_2 = expectation(model, dim2)

    E_12 = if dim1 == dim2
        weight_func = x -> x^2
        marginalize(model, dim1, weight_func)
    elseif dim1 > dim2
        weight_func = x -> x
        model = marginalize(model, dim1, weight_func)
        marginalize(model, dim2, weight_func)
    else
        weight_func = x -> x
        model = marginalize(model, dim2, weight_func)
        marginalize(model, dim1, weight_func)
    end
    if !((dim1 == dim2 && d==1) || (dim1 != dim2 && d==2))
        E_12 = marginalize(E_12)
    end
    return E_12 - E_1 * E_2
end

function covariance(model::PSDModelOrthonormal{d, T}) where {d, T<:Number}
    cov = zeros(T, d, d)
    for i in 1:d
        for j in i:d
            cov[i, j] = covariance(model, i, j)
        end
    end
    return Symmetric(cov)
end

## statistic functions for samplers

function expectation(sar::CondSampler{d, <:Any, T}; N_order=50) where {d, T<:Number}
    pdf_func = x->pdf(sar, x)
    x_gl, w_gl = gausslegendre(N_order)
    L = domain_interval_left(sar.models[1])
    R = domain_interval_right(sar.models[1])
    Xmat_gl = x_gl * ((R - L)./2)'
    Xmat_gl .+= ones(T, N_order) * ((R + L)./2)'
    corr_factor = prod((R - L)./2)
    gl_points(indices) = [Xmat_gl[indices[i], i] for i=1:d]
    gl_weights(indices) = mapreduce(i->w_gl[indices[i]],*,1:d)
    return corr_factor * mapreduce(i->gl_points(i)*gl_weights(i)*pdf_func(gl_points(i)),
                    +,
                    Iterators.product([1:N_order for _=1:d]...)
                    )
end


function covariance(sar::CondSampler{d, <:Any, T},
                dim1::Int, dim2::Int; N_order=50) where {d, T<:Number}
    E = expectation(sar)
    E_1 = E[dim1]
    E_2 = E[dim2]

    pdf_func = x->pdf(sar, x)
    x_gl, w_gl = gausslegendre(N_order)
    L = domain_interval_left(sar.models[1])
    R = domain_interval_right(sar.models[1])
    Xmat_gl = x_gl * ((R - L)./2)'
    Xmat_gl .+= ones(T, N_order) * ((R + L)./2)'
    corr_factor = prod((R - L)./2)
    gl_points(indices) = [Xmat_gl[indices[i], i] for i=1:d]
    gl_points(indices, dim) = Xmat_gl[indices[dim], dim]
    gl_weights(indices) = mapreduce(i->w_gl[indices[i]],*,1:d)
    E_12 = corr_factor * mapreduce(i->gl_weights(i)*gl_points(i,dim1)*gl_points(i,dim2)*pdf_func(gl_points(i)),
                    +,
                    Iterators.product([1:N_order for _=1:d]...)
                    )
    return E_12 - E_1 * E_2
end


function covariance(sar::CondSampler{d, <:Any, T}; N_order=50) where {d, T<:Number}
    E_X = expectation(sar)

    pdf_func = x->pdf(sar, x)
    x_gl, w_gl = gausslegendre(N_order)
    L = domain_interval_left(sar.models[1])
    R = domain_interval_right(sar.models[1])
    Xmat_gl = x_gl * ((R - L)./2)'
    Xmat_gl .+= ones(T, N_order) * ((R + L)./2)'
    corr_factor = prod((R - L)./2)
    gl_points(indices) = [Xmat_gl[indices[i], i] for i=1:d]
    gl_weights(indices) = mapreduce(i->w_gl[indices[i]],*,1:d)
    E_XY = corr_factor * mapreduce(i->gl_weights(i)*(gl_points(i)*gl_points(i)')*pdf_func(gl_points(i)),
                    +,
                    Iterators.product([1:N_order for _=1:d]...)
                    )
    return E_XY - (E_X * E_X')
end


end # module Statistics
