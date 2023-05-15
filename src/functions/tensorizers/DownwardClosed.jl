

"""
    DownwardClosedTensorizer{d}
A downward closed set is defined by

"""
struct DownwardClosedTensorizer{d} <: Tensorizer{d}
    index_list::Vector{Vector{Int}}
    M_inner::Vector{Vector{Int}} # inner Margin of downward_closed set
end

function DownwardClosedTensorizer(d::Int, max_order::Int)
    index_list = Vector{Int}[];
    for i=0:max_order
        vec = map(i->i.+1, collect(multiexponents(d, i)))
        push!(index_list, vec...)
    end
    inner_margin = map(i->i.+1, collect(multiexponents(d, max_order)))
    return DownwardClosedTensorizer{d}(index_list, inner_margin)
end

# function DownwardClosedTensorizer(max_element_list::Vector{Int})
#     d = length(max_element_list)
#     ten = DownwardClosedTensorizer(d, max(max_element_list...))
#     new_index_list = filter(x -> all(i->i<0, x.-max_element_list), ten.index_list)
#     return DownwardClosedTensorizer{d}(new_index_list, max_element_list)
# end

@inline _valid_index(index::Vector{Int}) = all(i->i>0, index)

function check_in_outer_margin(t::DownwardClosedTensorizer{d}, 
        index::Vector{Int}) where {d}
    @assert length(index) == d
    delta(i) = begin
        a = zeros(Int, d)
        a[i] = 1
        return a
    end
    if index in t.index_list
        return false
    end
    for i=1:d
        i_tmp = index-delta(i)
        if _valid_index(i_tmp) && !(i_tmp in t.index_list)
            return false
        end
    end
    return true
end

function check_in_inner_margin(t::DownwardClosedTensorizer{d}, 
        index::Vector{Int}) where {d}
    @assert length(index) == d
    delta(i) = begin
        a = zeros(Int, d)
        a[i] = 1.0
        return a
    end
    if !(index in t.index_list)
        return false
    end
    for i=1:d
        if !((index+delta(i)) in t.index_list)
            return true
        end
    end
    return false
end

function inner_margin_neighbors(t::DownwardClosedTensorizer{d},
    outer_margin::Vector{Int}) where {d}
    delta(i) = begin
        a = zeros(Int, d)
        a[i] = 1
        return a
    end
    downward_elements = [outer_margin-delta(i) for i=1:d]
    list = [x for x in t.M_inner if x in downward_elements]
    return list
end

function add_index!(t::DownwardClosedTensorizer{d}, index::Vector{Int}) where {d}
    @assert length(index) == d
    @assert check_in_outer_margin(t, index)
    push!(t.index_list, index)
    # marg_neighbors = inner_margin_neighbors(t, index)
    push!(t.M_inner, index)
    filter!(x->check_in_inner_margin(t, x), t.M_inner)
    return nothing
end

### Functions common to all Tensorizers

@inline σ(t::DownwardClosedTensorizer, i) = @error "not implemented for this tensorizer"
@inline σ_inv(t::DownwardClosedTensorizer, i) = t.index_list[i]
@inline max_N(t::DownwardClosedTensorizer) = length(t.index_list)

function reduce_dim(t::DownwardClosedTensorizer{d}, dim::Int) where {d}
    @assert 1≤dim≤d
    M_inner = unique(map(x->x[1:end .!= dim], t.M_inner))
    reduced_index_list = unique(map(i->i[1:end .!= dim], t.index_list))
    ten = DownwardClosedTensorizer{d-1}(reduced_index_list, M_inner)
    filter!(x->check_in_inner_margin(ten, x), ten.M_inner)
    return ten
end

function highest_order(t::DownwardClosedTensorizer)
    return max([sum(x) for x in t.M_inner]...)
end

function next_index_proposals(t::DownwardClosedTensorizer{d}) where {d}
    delta(i) = begin
        a = zeros(Int, d)
        a[i] = 1
        return a
    end
    proposals = [x + delta(i) for x in t.M_inner for i=1:d]
    reshape(proposals, length(proposals))
    filter!(x->check_in_outer_margin(t, x), proposals)
    return unique(proposals)
end