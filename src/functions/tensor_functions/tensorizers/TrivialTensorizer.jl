
struct TrivialTensorizer{d} <: Tensorizer{d}
    order::AbstractArray{Int, d}
    CI::CartesianIndices{d}
    function TrivialTensorizer{d}(
                order::AbstractArray{Int, d}, 
                CI::CartesianIndices{d}) where {d}
        return new{d}(order, CI)
    end
end

### FUnctions specific to TrivialTensorizer

function TrivialTensorizer(r_list::Vector{Int})
    d = length(r_list)
    vec = collect(1:prod(r_list))
    order = @views reshape(vec[:], [r_list[i] for i=1:d]...)
    CI = CartesianIndices(order)
    return TrivialTensorizer{d}(order, CI)
end

function TrivialTensorizer(d::Int, N::Int)
    highest_order = Int(ceil(N^(1/d)))
    vec = collect(1:highest_order^d)
    order = @views reshape(vec[:], [highest_order for i=1:d]...)
    CI = CartesianIndices(order)
    return TrivialTensorizer{d}(order, CI)
end

### Functions common to all Tensorizers

@inline σ(t::TrivialTensorizer, i) = t.order[i...]
@inline σ_inv(t::TrivialTensorizer, i::Int) = Tuple(t.CI[i])
@inline max_N(t::TrivialTensorizer) = prod(size(t.order))

function reduce_dim(t::TrivialTensorizer{d}, dim::Int) where {d}
    highest_order = size(t.order, 1)
    N = highest_order^(d-1)
    return TrivialTensorizer(d-1, N)
end

function highest_order(t::TrivialTensorizer)
    return max(size(t.order)...)
end

function add_order(t::TrivialTensorizer{d}, dim::Int) where {d}
    r_list = collect(size(t.order))
    r_list[dim] += 1
    return TrivialTensorizer(r_list)
end

function permute_indices(t::TrivialTensorizer{d}, perm::Vector{Int}) where {d}
    if d>1 
        throw(ArgumentError("Not implemented for d>1, convert to DownwardClosedTensorizer first"))
        order_new = permutedims(t.order, perm)
        # CI_new = CartesianIndices(order_new)
        return TrivialTensorizer{d}(order_new, CI)
    else
        return t
    end
end

function check_in_tensorizer(t::TrivialTensorizer{d}, 
    index::NTuple{d, Int}) where {d}
    throw(ArgumentError("Not implemented for d>1, convert to DownwardClosedTensorizer first"))
    return CartesianIndex(index) in t.CI
end
