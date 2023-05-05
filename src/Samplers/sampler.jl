
abstract type Sampler{T} end

Sampler(model::PSDModel) = @error "not implemented for this type of PSDModel"

include("PSDModelSampler.jl")