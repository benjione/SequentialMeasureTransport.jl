

struct DownwardClosedTensorizer{d} <: Tensorizer{d}
    index_list::Vector{Vector{Int}}
    max_element_list::Vector{Int}
end

function DownwardClosedTensorizer(d::Int, max_order::Int)
    index_list = Vector{Int}[];
    for i=0:max_order
        vec = map(i->i.+1, collect(multiexponents(d, i)))
        push!(index_list, vec...)
    end
    max_element_list = max_order * ones(Int, d)
    return DownwardClosedTensorizer{d}(index_list, max_element_list)
end

function DownwardClosedTensorizer(max_element_list::Vector{Int})
    d = length(max_element_list)
    ten = DownwardClosedTensorizer(d, max(max_element_list...))
    new_index_list = filter(x -> all(i->i<0, x.-max_element_list), ten.index_list)
    return DownwardClosedTensorizer{d}(new_index_list, max_element_list)
end

function add_index(t::DownwardClosedTensorizer{d}, index::Vector{Int}) where {d}
    @assert length(index) == d
    vec = index.-t.max_element_list
    vec[vec .< 0] = 0
    max_element_list = copy(t.max_element_list)
    max_element_list .+= vec
    new_index_list = copy(t.index_list)
    push!(new_index_list, index)
    return DownwardClosedTensorizer{d}(new_index_list, max_element_list)
end

### Functions common to all Tensorizers

@inline σ(t::DownwardClosedTensorizer, i) = @error "not implemented for this tensorizer"
@inline σ_inv(t::DownwardClosedTensorizer, i) = t.index_list[i]
@inline max_N(t::DownwardClosedTensorizer) = length(t.index_list)

function reduce_dim(t::DownwardClosedTensorizer{d}, dim::Int) where {d}
    @assert 1≤dim≤d
    max_element_list = t.max_element_list[1:end .!= dim]
    reduced_index_list = unique(map(i->i[1:end .!= dim], t.index_list))
    return DownwardClosedTensorizer{d-1}(reduced_index_list, max_element_list)
end

function highest_order(t::DownwardClosedTensorizer)
    return max(t.max_element_list...)
end