

struct GridMapping{d, dC, T} <: AbstractTriangularMap{d, dC, T}
    Δ::Vector{T}
    grid_values::AbstractArray{T, d}
    marginal_grids::Vector{AbstractArray{T}}
    integrated_grids::Vector{AbstractArray{T}}
    variable_ordering::Vector{Int}
    in_dom::Vector{Tuple{T, T}}
    out_dom::Vector{Tuple{T, T}}
    function GridMapping(grid_values::AbstractArray{T, d}, Δ::Vector{T}) where {d, T}
        variable_ordering = collect(1:d)
        _grid_val = permutedims(grid_values, variable_ordering)
        marginal_grids = [prod(Δ[(i+1):d]) * sum(_grid_val, dims=(i+1):d) for i in 1:d]
        function integr(A::AbstractArray{T, _d}, dim::Int, Δ::T, marg_val::T) where {_d}
            s = size(A)
            s = ntuple(i->if i==dim
                s[i]
            else
                s[i]
            end, _d)
            B = zeros(T, s)
            selec1 = ntuple(i->if i==dim
                2:s[i]
            else
                1:s[i]
            end, _d)
            selec2 = ntuple(i->if i==dim
                1:s[i]-1
            else
                1:s[i]
            end, _d)
            B[selec1...] = Δ*cumsum(A, dims=dim)[selec2...]
            selec = ntuple(i->if i==dim
                s[i]
            else
                1
            end, _d)
            B *= marg_val / B[selec...]
            return B
        end
        integrated_grids = [integr(marginal_grids[i], i, Δ[i], 
                    i-1 < 1 ? 1.0 : marginal_grids[i-1][ones(Int, i-1)...]) for i in 1:d]
        # integrated_grids = [Δ[i] * cumsum(marginal_grids[i], dims=i) for i in 1:d]
        in_dom = [(0.0, 1.0) for _ in 1:d]
        out_dom = [(0.0, 1.0) for _ in 1:d]
        new{d, 0, T}(Δ, grid_values, marginal_grids, integrated_grids, 
                variable_ordering, in_dom, out_dom)
    end
end


function MonotoneMap(map::GridMapping{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number}
    lower_entries = zeros(Int, k)
    ex_entry = zero(T)
    for i in 1:(k)
        ex_entry = x[i] / map.Δ[i]
        lower_entries[i] = Int(floor(ex_entry)) + 1
    end
    marg_value = if k==1
        1.0
    else
        map.marginal_grids[k-1][lower_entries[1:end-1]...]
    end
    a = map.integrated_grids[k][lower_entries[1:end]...]
    b = if lower_entries[end] == size(map.grid_values)[k]
        a
    else
        map.integrated_grids[k][lower_entries[1:end-1]..., lower_entries[end] + 1]
    end
    # b = map.integrated_grids[k][lower_entries[1:end-1]..., lower_entries[end] + 1]
    return (a + (x[k] - (lower_entries[k]-1)* map.Δ[k])/map.Δ[k] * (b - a)) / marg_value
end

function ∂k_MonotoneMap(map::GridMapping{d, <:Any, T}, x::PSDdata{T}, k::Int) where {d, T<:Number}
    lower_entries = zeros(Int, k)
    for i in 1:(k)
        lower_entries[i] = floor(x[i] / map.Δ[i]) + 1
    end
    marg_value = if k==1
        1.0
    else
        map.marginal_grids[k-1][lower_entries[1:end-1]...]
    end
    a = map.integrated_grids[k][lower_entries[1:end]...]
    b = if lower_entries[end] == size(map.grid_values)[k]
        a
    else
        map.integrated_grids[k][lower_entries[1:end-1]..., lower_entries[end] + 1]
    end
    b = map.integrated_grids[k][lower_entries[1:end-1]..., lower_entries[end] + 1]
    return (b - a) / (marg_value * map.Δ[k])
end