
abstract type Tensorizer{d} end

include("TrivialTensorizer.jl")


### Interface for Tensorizers
@inline σ(t::Tensorizer, i) = @error "not implemented for this tensorizer"
@inline σ_inv(t::Tensorizer, i) = @error "not implemented for this tensorizer"
@inline max_N(t::Tensorizer) = @error "not implemented for this tensorizer"

function reduce_dim(t::Tensorizer{d}) where {d}
    @error "not implemented for this tensorizer"
end

function highest_order(t::Tensorizer)
    @error "not implemented for this tensorizer"
end
