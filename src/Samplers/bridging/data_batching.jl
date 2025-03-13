

struct DataBatchingBridgingDensity{d, T} <: BridgingDensity{d, T}
    f::Function     # Forward function from parameterspace X to Y
    data            # in Y
    U_list          # list of unitary matrices for each bridging density, with U_1 ⊂ U_2 ⊂ ...
    σ
end


function evaluate_bridge(bridge::DataBatchingBridgingDensity{d, T}, x::PSDdata{T}, i) where {T<:Number}
    U = bridge.U_list[i]
    exp(-norm(U * bridge.f(x) - U * bridge.data, 2)^2 / 2*bridge.σ^2)
end