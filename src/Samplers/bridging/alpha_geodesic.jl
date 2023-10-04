

struct AlphaGeodesicBridgingDensity{α, d, T} <: BridgingDensity{d, T}
    start_measure::Function
    end_measure::Function
    t_list::Vector{T}
    broadcasting::Bool          # if true, end measure is a broadcast function
    function AlphaGeodesicBridgingDensity{d}(α::T,
                                             start_measure::Function, 
                                             end_measure::Function,
                                             t_list::Vector{T};
                                             broadcasting=false,
                                             dual=false) where {d, T<:Number}
        if dual # use dual geodesic
            α = 1 - α
        end
        new{α, d, T}(start_measure, end_measure, t_list, broadcasting)
    end
end

function evaluate_bridge(bridge::AlphaGeodesicBridgingDensity{α, d, T}, 
                        x::PSDdata{T}, 
                        k::Int) where {α, d, T<:Number}
    return evaluate_bridge(bridge, x, bridge.t_list[k])
end

function evaluate_bridge(bridge::AlphaGeodesicBridgingDensity{α, d, T}, 
                        x::PSDdata{T}, 
                        t::T) where {α, d, T<:Number}
    if α == 1
        return bridge.start_measure(x)^(1 - t) * bridge.end_measure(x)^t
    end
    return  (
                (1 - t) * bridge.start_measure(x)^(1 - α) + 
                t * bridge.end_measure(x)^(1 - α)
            )^(1 / (1 - α))
end