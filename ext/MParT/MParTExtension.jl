module MParTExtension

import SequentialMeasureTransport as SMT
using MParT


struct MParTMap{d, dC, T} <: SMT.AbstractCondSampler{d,dC,T,Nothing,Nothing}
    map
    function MParTMap{d, dC, T}(map) where {d, dC, T}
        new{d, dC, T}(map)
    end
    function MParTMap(InOutDim::Int, total_order::Int, map_opts::MParT.MapOptions) where {d, dC, T}
        map = CreateTriangular(InOutDim, InOutDim, total_order, map_opts)
        new{d, dC, T}(map)
    end
end

function SMT.pushforward(sra::MParTMap{d, dC, T}, u::PSDdata{T}) where {d, dC, T}
    return MParT.evaluate(sra.map, u)
end

function SMT.pullback(sra::MParTMap{d, dC, T}, u::PSDdata{T}) where {d, dC, T}
    return MParT.inverse(sra.map, u)
end

end