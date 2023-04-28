module Statistics

using ..PSDModels
using ..PSDModels: PSDDataVector

export ML_fit!, Chi2_fit!

"""
    ML_fit!(model, samples; kwargs...)

Maximum likelihood fit of a PSD model to the samples.
"""
function ML_fit!(model::PSDModel{T}, 
    samples::PSDDataVector{T};
    kwargs...) where {T<:Number}

    loss_KL(Z) = -(1/length(Z)) * sum(log.(Z))
    minimize!(model, loss_KL, samples; 
            normalization_constraint=true,
            kwargs...)
end


function Chi2_fit!(model::PSDModel{T}, 
    X::PSDDataVector{T},
    Y::PSDDataVector{T};
    ϵ=1e-5,
    kwargs...) where {T<:Number}

    # Chi2 defined by ∫ (f(x) - y(x))^2/y(x) dx
    # => IRLS with weights 1/(y(x) + ϵ), ϵ for numerical reasons

    # Reweighting of the IRLS algorithm
    reweight(z) = 1 / (z + ϵ)
 
    IRLS!(model, X, Y, reweight; kwargs...)
end

end # module Statistics
