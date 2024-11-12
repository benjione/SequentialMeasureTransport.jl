
abstract type Tensorizer{d} end

include("TrivialTensorizer.jl")
include("DownwardClosed.jl")


### Interface for Tensorizers
@inline σ(t::Tensorizer, i) = @error "not implemented for this tensorizer"
@inline σ_inv(t::Tensorizer, i) = @error "not implemented for this tensorizer"
@inline max_N(t::Tensorizer) = @error "not implemented for this tensorizer"

function reduce_dim(t::Tensorizer{d}, dim::Int) where {d}
    @error "not implemented for this tensorizer"
end

function highest_order(t::Tensorizer)
    @error "not implemented for this tensorizer"
end

function next_index_proposals(t::Tensorizer)
    @error "not implemented for this tensorizer"
end

function add_index!(t::Tensorizer, index::Vector{Int})
    @error "not implemented for this tensorizer"
end

function permute_indices(t::Tensorizer, perm::Vector{Int})
    @error "not implemented for this tensorizer"
end



"""
Conversion of Tensorizer
"""
function trivial_to_downward_closed(t::TrivialTensorizer{d}) where {d}
    index_list = t.CI[1:end] .|> x->Tuple(x)
    return DownwardClosedTensorizer(index_list)
end