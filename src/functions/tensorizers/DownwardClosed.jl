

struct DownwardClosedTensorizer{d} <: Tensorizer{d}
    index_list::AbstractVector{Vector{Int}}
    highest_order_list::AbstractVector{Int}
end

function DownwardClosedTensorizer(d::Int, max_order::Int)
    index_list = Vector{Int}[];
    for i=1:max_order
        push!(index_list, colllect(multiexponents(i,d)))
    end
    highest_order_list = max_order * ones(Int, d)
    return DownwardClosedTensorizer{d}(index_list, highest_order_list)
end

function DownwardClosedTensorizer(highest_order_list::AbstractVector{Int})
    d = length(highest_order_list)
    index_list = Vector{Int}[];
    max_order = max(highest_order_list...)
    for i=1:max_order
        push!(index_list, colllect(multiexponents(i,d)))
    end
    filter!(x -> all(i->i<0, x.-highest_order_list), index_list)
    return DownwardClosedTensorizer{d}(index_list, highest_order_list)
end

### Functions common to all Tensorizers

@inline σ(t::DownwardClosedTensorizer, i) = @error "not implemented for this tensorizer"
@inline σ_inv(t::DownwardClosedTensorizer, i) = t.index_list[i]
@inline max_N(t::DownwardClosedTensorizer) = length(t.index_list)

function reduce_dim(t::DownwardClosedTensorizer{d}, dim::Int) where {d}
    @assert 1≤dim≤d
    highest_order_list = t.highest_order_list[1:end .!= dim]
    return DownwardClosedTensorizer(highest_order_list)
end

function highest_order(t::DownwardClosedTensorizer)
    return max(t.highest_order_list...)
end