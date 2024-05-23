

"""
Composition of other reference maps.

Composition is applied in inverse order of components:
    T = T_1 ◦ T_2 ◦ ... ◦ T_n
where T_i is the i-th reference map in components.
"""
struct ComposedReference{d, dC, T} <: ReferenceMap{d, dC, T} 
    components::Vector{<:ReferenceMap{d, dC, T}}
    function ComposedReference{d, dC, T}(components::Vector{<:ReferenceMap{d, dC, T}}) where {d, dC, T<:Number}
        new{d, dC, T}(components)
    end
end


"""
    normalized_gaussian_reference(mean, std)

Reference map for normalizing data in R^d with Gaussian reference.

# Arguments
- `mean`: The mean of the data.
- `std`: The standard deviation of the data.

# Returns
A `ComposedReference` object that represents the composed reference map.

# Examples
data_mean = mean.(data)
data_std = std.(data)
R1 = normalized_gaussian_reference(data_mean, data_std)
"""
normalized_gaussian_reference(mean, std) = normalized_gaussian_reference(mean, std, 0)
function normalized_gaussian_reference(mean::Vector{T}, std::Vector{T}, dC::Int) where {T<:Number}
    d = length(mean)
    gauss_component = GaussianReference{d, dC, T}()
    scaling_component = NormalizationReference(mean, std, dC)
    ComposedReference{d, dC, T}([gauss_component, scaling_component])
end

"""
    normalized_algebraic_reference(mean, std)

As normalized_gaussian_reference with algebraic map instead of Gaussian.
"""
normalized_algebraic_reference(mean, std) = normalized_algebraic_reference(mean, std, 0)
function normalized_algebraic_reference(mean::Vector{T}, std::Vector{T}, dC::Int) where {T<:Number}
    d = length(mean)
    alg_map = AlgebraicReference{d, dC, T}()
    scaling_component = NormalizationReference(mean, std, dC)
    ComposedReference{d, dC, T}([alg_map, scaling_component])
end

function SMT.pushforward(
        m::ComposedReference{d, <:Any, T}, 
        x::PSDdata{T}
    ) where {d, T<:Number}
    for component in reverse(m.components)
        x = SMT.pushforward(component, x)
    end
    return x
end

function SMT.pullback(
        m::ComposedReference{d, <:Any, T}, 
        u::PSDdata{T}
    ) where {d, T<:Number}
    for component in m.components
        u = SMT.pullback(component, u)
    end
    return u
end

function SMT.pushforward(
        m::ComposedReference{d, <:Any, T}, 
        π::Function
    ) where {d, T<:Number}
    for component in reverse(m.components)
        π = SMT.pushforward(component, π)
    end
    return π
end

function SMT.pullback(
        m::ComposedReference{d, <:Any, T}, 
        π::Function
    ) where {d, T<:Number}
    for component in m.components
        π = SMT.pullback(component, π)
    end
    return π
end

function SMT.marginal_pushforward(
        m::ComposedReference{d, dC, T}, 
        x::PSDdata{T}
    ) where {d, dC, T<:Number}
    for component in reverse(m.components)
        x = SMT.marginal_pushforward(component, x)
    end
    return x
end

function SMT.marginal_pullback(
        m::ComposedReference{d, dC, T}, 
        u::PSDdata{T}
    ) where {d, dC, T<:Number}
    for component in m.components
        u = SMT.marginal_pullback(component, u)
    end
    return u
end

function SMT.marginal_pushforward(
        m::ComposedReference{d, <:Any, T}, 
        π::Function
    ) where {d, T<:Number}
    for component in reverse(m.components)
        π = SMT.marginal_pushforward(component, π)
    end
    return π
end

function SMT.marginal_pullback(
        m::ComposedReference{d, <:Any, T}, 
        π::Function
    ) where {d, T<:Number}
    for component in m.components
        π = SMT.marginal_pullback(component, π)
    end
    return π
end

function SMT.log_pushforward(
        m::ComposedReference{d, <:Any, T}, 
        π::Function
    ) where {d, T<:Number}
    for component in reverse(m.components)
        π = SMT.log_pushforward(component, π)
    end
    return π
end

function SMT.log_pullback(
        m::ComposedReference{d, <:Any, T}, 
        π::Function
    ) where {d, T<:Number}
    for component in m.components
        π = SMT.log_pullback(component, π)
    end
    return π
end

function SMT.marginal_log_pushforward(
        m::ComposedReference{d, <:Any, T}, 
        π::Function
    ) where {d, T<:Number}
    for component in reverse(m.components)
        π = SMT.marginal_log_pushforward(component, π)
    end
    return π
end

function SMT.marginal_log_pullback(
        m::ComposedReference{d, <:Any, T}, 
        π::Function
    ) where {d, T<:Number}
    for component in m.components
        π = SMT.marginal_log_pullback(component, π)
    end
    return π
end

function SMT.Jacobian(sra::ComposedReference{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number}
    res = one(T)
    for component in reverse(sra.components)
        res *= SMT.Jacobian(component, x)
        x = SMT.pushforward(component, x)
    end
    return res
end

function SMT.inverse_Jacobian(sra::ComposedReference{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number}
    res = one(T)
    for component in sra.components
        res *= SMT.inverse_Jacobian(component, x)
        x = SMT.pullback(component, x)
    end
    return res
end


@inline SMT.log_Jacobian(sra::ComposedReference{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number} = SMT.log_pullback(sra, x->zero(T))(x)
@inline SMT.inverse_log_Jacobian(sra::ComposedReference{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number} = SMT.log_pushforward(sra, x->zero(T))(x)

@inline SMT.marginal_inverse_Jacobian(sra::ComposedReference{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number} = SMT.marginal_pushforward(sra, x->one(T))(x)
@inline SMT.marginal_Jacobian(sra::ComposedReference{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number} = SMT.marginal_pullback(sra, x->one(T))(x)


@inline SMT.marginal_log_Jacobian(sra::ComposedReference{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number} = SMT.marginal_log_pullback(sra, x->zero(T))(x)
@inline SMT.marginal_inverse_log_Jacobian(sra::ComposedReference{<:Any, <:Any, T}, x::PSDdata{T}) where {T<:Number} = SMT.marginal_log_pushforward(sra, x->zero(T))(x)
