
struct TrivialTensorizer{d} <: Tensorizer{d}
    σ::Function
    σ_inv::Function
    order::Array{Int, d}
    CI::CartesianIndices{d}
    function TrivialTensorizer(d::Int, N::Int)
        highest_order = Int(ceil(N^(1/d)))
        vec = collect(1:highest_order^d)
        order = @views reshape(vec[:], [highest_order for i=1:d]...)
        CI = CartesianIndices(order)
        return new{d}(σ, σ_inv, order, CI)
    end
end

@inline σ(t::TrivialTensorizer, i) = t.order[i...]
@inline σ_inv(t::TrivialTensorizer, i) = t.CI[i]

"""
Return a TrivialTensorizer with one dimension less.
"""
function reduce_dim(t::TrivialTensorizer{d}) where {d}
    highest_order = size(t.order, 1)
    N = highest_order^(d-1)
    return TrivialTensorizer(d-1, N)
end

function highest_order(t::TrivialTensorizer)
    return size(t.order, 1)
end
