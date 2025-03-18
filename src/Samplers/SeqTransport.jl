

struct SeqTransport{d, dC, T}
    target_density::Function
    bridging_density::BridgingDensity{d, T}
    sample_list::AbstractVector{Tuple{<:PSDDataVector{T}, Vector{T}}}
    ref_map::ReferenceMap
    is_next_fit_nexessary
    create_next_fit!
end

function SeqTransport{d, dC, T}(target_density, bridging_density, 
                    ref_map, is_next_fit_nexessary, 
                    create_next_fit!) where {d, dC, T}

    sample_list = Tuple{<:PSDDataVector{T}, Vector{T}}[]
    SeqTransport{d, dC, T}(target_density, bridging_density, 
            sample_list, ref_map, is_next_fit_nexessary, create_next_fit!)
end


function construct_all(a::SeqTransport{d, dC, T}) where {d, dC, T}

    seqSampler = CondSampler(ConditionalMapping{d, dC, T}[], a.ref_map)

    i = 0
    while(a.is_next_fit_nexessary(i, a.bridging_density, seqSampler))
        i += 1
        map = a.create_next_fit!(a.sample_list, a.bridging_density, a.ref_map, seqSampler, i)
        push!(seqSampler.samplers, map)
    end
    return seqSampler
end