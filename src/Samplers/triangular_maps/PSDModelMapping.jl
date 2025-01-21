

struct PSDModelMapping{d, dC, T} <: AbstractTriangularMap{d, dC, T}
    model::PSDModel{T}
    margins::Vector{<:PSDModel{T}}
    integrals::Vector{<:TraceModel{T}}
    variable_ordering::Vector{Int}
    out_dom::Vector{Tuple{<:Number, <:Number}}
    function PSDModelMapping(model::PSDModel{T}, 
                    variable_ordering::Vector{Int},
                    dC::Int) where {T}
        d = dimension(model)
        out_dom = [(-Inf, Inf) for k in 1:d]
        model = normalize(model)
        perm_model = permute_indices(model, variable_ordering)
        margins = PSDModel{T}[marginalize(perm_model, collect(k:d)) for k in 2:d]
        push!(margins, perm_model)
        integrals = map((x,k)->integral(x, k), margins, 1:d)
        new{d, dC, T}(model, margins, integrals, variable_ordering, out_dom)
    end
end


@inline MonotoneMap(sampler::PSDModelMapping{d, <:Any, T}, 
            x::PSDdata{T}, 
            k::Int) where {d, T<:Number} = begin
    if k==1
        return sampler.integrals[k](x)
    end
    return (sampler.integrals[k](x)/sampler.margins[k-1](x[1:k-1]))
end
@inline ∂k_MonotoneMap(sampler::PSDModelMapping{d, <:Any, T}, 
            x::PSDdata{T}, 
            k::Int) where {d, T<:Number} = begin
    if k==1
        return sampler.margins[k](x)
    end
    return (sampler.margins[k](x)/sampler.margins[k-1](x[1:k-1]))
end

@inline Jacobian(sampler::PSDModelMapping{d, <:Any, T}, 
                 x::PSDdata{T}
        ) where {d, T<:Number} = sampler.model(x)


function marginal_Jacobian(sampler::PSDModelMapping{d, dC, T}, x::PSDdata{T}) where {d, dC, T<:Number}
    return sampler.margins[d-dC](x)
end