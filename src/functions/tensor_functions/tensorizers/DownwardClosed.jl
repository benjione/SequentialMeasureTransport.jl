

"""
    DownwardClosedTensorizer{d}
A downward closed set is defined by

"""
struct DownwardClosedTensorizer{d} <: Tensorizer{d}
    index_list::Vector{NTuple{d, Int}}
    M_inner::Vector{NTuple{d, Int}} # inner Margin of downward_closed set
end

function DownwardClosedTensorizer(index_list::Vector{NTuple{d, Int}}) where {d}
    M_inner = NTuple{d, Int}[]
    ten = DownwardClosedTensorizer{d}(index_list, M_inner)
    M_inner = filter(x->check_in_inner_margin(ten, x), index_list)
    for i in M_inner
        push!(ten.M_inner, i)
    end
    return ten
end

function DownwardClosedTensorizer(d::Int, max_order::Int; max_Φ_size=nothing)
    index_list = NTuple{d, Int}[];
    for i=0:max_order
        vec = map(i->i.+1, collect(multiexponents(d, i)))
        if max_Φ_size !== nothing
            if length(vec) + length(index_list) > max_Φ_size
                vec = StatsBase.sample(vec, max_Φ_size-length(index_list); replace=false)
            end
        end
        push!(index_list, Tuple.(vec)...)
    end
    inner_margin = map(i->i.+1, collect(multiexponents(d, max_order)))
    return DownwardClosedTensorizer{d}(index_list, Tuple.(inner_margin))
end


function permute_indices(t::DownwardClosedTensorizer{d}, perm::Vector{Int}) where {d}
    perm_tuple(tp, perm) = Tuple([tp...][perm])
    index_list = map(i->perm_tuple(i, perm), t.index_list)
    M_inner = map(i->perm_tuple(i, perm), t.M_inner)
    return DownwardClosedTensorizer{d}(index_list, M_inner)
end

@inline _valid_index(index::NTuple{<:Any, Int}) = all(i->i>0, index)

function check_in_tensorizer(t::DownwardClosedTensorizer{d}, 
        index::NTuple{d, Int}) where {d}
    return index in t.index_list
end

function check_in_outer_margin(t::DownwardClosedTensorizer{d}, 
        index::NTuple{d, Int}) where {d}
    delta(i) = begin
        a = zeros(Int, d)
        a[i] = 1
        return a
    end
    if index in t.index_list
        return false
    end
    for i=1:d
        i_tmp = Tuple([index...]-delta(i))
        if _valid_index(i_tmp) && !(i_tmp in t.index_list)
            return false
        end
    end
    return true
end

function check_in_inner_margin(t::DownwardClosedTensorizer{d}, 
        index::NTuple{d, Int}) where {d}
    delta(i) = begin
        a = zeros(Int, d)
        a[i] = 1.0
        return a
    end
    if !(index in t.index_list)
        return false
    end
    for i=1:d
        if !(Tuple([index...]+delta(i)) in t.index_list)
            return true
        end
    end
    return false
end

function inner_margin_neighbors(t::DownwardClosedTensorizer{d},
    outer_margin::NTuple{d, Int}) where {d}
    delta(i) = begin
        a = zeros(Int, d)
        a[i] = 1
        return a
    end
    downward_elements = [Tuple([outer_margin...]-delta(i)) for i=1:d]
    list = [x for x in t.M_inner if x in downward_elements]
    return list
end

add_index!(t::DownwardClosedTensorizer{d}, index::Vector{Int}) where {d} = begin
    @assert length(index) == d
    add_index!(t, Tuple(index))
end
function add_index!(t::DownwardClosedTensorizer{d}, index::NTuple{d, Int}) where {d}
    @assert check_in_outer_margin(t, index)
    push!(t.index_list, index)
    # marg_neighbors = inner_margin_neighbors(t, index)
    push!(t.M_inner, index)
    filter!(x->check_in_inner_margin(t, x), t.M_inner)
    return nothing
end

### Functions common to all Tensorizers

@inline σ(t::DownwardClosedTensorizer, i) = @error "not implemented for this tensorizer"
@inline σ_inv(t::DownwardClosedTensorizer, i::Int) = t.index_list[i]
@inline max_N(t::DownwardClosedTensorizer) = length(t.index_list)

function reduce_dim(t::DownwardClosedTensorizer{d}, dim::Int) where {d}
    @assert 1≤dim≤d
    M_inner = unique(map(x->x[1:end .!= dim], t.M_inner))
    reduced_index_list = unique(map(i->i[1:end .!= dim], t.index_list))
    ten = DownwardClosedTensorizer{d-1}(reduced_index_list, M_inner)
    filter!(x->check_in_inner_margin(ten, x), ten.M_inner)
    return ten
end

function tensorize(t1::DownwardClosedTensorizer{d}, t2::DownwardClosedTensorizer{e}) where {d, e}
    it = Iterators.product(t1.index_list, t2.index_list)
    it2 = Iterators.product(t1.M_inner, t2.M_inner)
    index_list = [(x..., y...) for (x, y) in it]
    M_inner = [(x..., y...) for (x,y) in it2 ]
    index_list = reshape(index_list, length(index_list))
    M_inner = reshape(M_inner, length(M_inner))
    return DownwardClosedTensorizer{d+e}(index_list, M_inner)
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
    proposals = [Tuple([x...] + delta(i)) for x in t.M_inner for i=1:d]
    reshape(proposals, length(proposals))
    filter!(x->check_in_outer_margin(t, x), proposals)
    return unique(proposals)
end