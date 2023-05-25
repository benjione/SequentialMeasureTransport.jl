module Plotting
"""
Submodule to create nice plots of the results and samplers.
"""

using PSDModels
using Plots

using ..PSDModels
using ..PSDModels: SelfReinforcedSampler


function plot_sampler2D(sar::SelfReinforcedSampler,
            target_list::Vector{Function};
            domain=nothing
        )
    domain = domain === nothing ? (-1.0, 1.0) : domain
    domx = range(domain[1], domain[2], length=100)
    plt_list_forward = []
    for (i, model) in enumerate(sar.models)
        PSDModels.pushforward_pdf_function(sar, target_list[i];layers=collect(1:i))
        push!(plt_list_forward, contour(domx, domx, (x,y)->model([x,y]), 
                title="model $i")
        )
    end
    plt_list_pullback = []
    for (i, tar) in enumerate(target_list)
        pullback_func = PSDModels.pullback_pdf_function(sar, tar;layers=collect(1:i-1))
        push!(plt_list_pullback, contour(domx, domx, (x,y)->pullback_func([x,y]), 
                title="pullback targets $i")
        )
    end
    return plot(plt_list_forward..., plt_list_pullback..., layout=(2, length(sar.models)))
end


end
