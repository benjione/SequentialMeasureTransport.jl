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
            domain=nothing,
            N_plot=100,
            kwargs...
        )
    @assert length(sar.models) == length(target_list)
    domain = domain === nothing ? (-1.0, 1.0) : domain
    domx = range(domain[1], domain[2], length=N_plot)
    plt_list_forward = []
    for (i, model) in enumerate(sar.models)
        pdf_func = PSDModels.pushforward_pdf_function(sar;layers=collect(1:i))
        push!(plt_list_forward, contour(domx, domx, (x,y)->pdf_func([x,y]), 
                title="\$ \\left( T_{\\leq $i} \\right)_{\\#} \\rho_{ref} \$"; kwargs...)
        )
    end
    plt_list_pullback = []
    for (i, tar) in enumerate(target_list)
        pullback_func = PSDModels.pullback_pdf_function(sar, tar;layers=collect(1:i-1))
        if i==1
            push!(plt_list_pullback, contour(domx, domx, (x,y)->pullback_func([x,y]), 
                    title="Target"; kwargs...))
        else
            push!(plt_list_pullback, contour(domx, domx, (x,y)->pullback_func([x,y]), 
                    title="\$T_{ \\leq $(i-1) }^\\# \\pi_{$i } \$";  kwargs...))
        end
    end
    plt_pulledback_reference = []
    for (i, tar) in enumerate(target_list)
        pullback_func = PSDModels.pullback_pdf_function(sar, tar;layers=collect(1:i))
        push!(plt_pulledback_reference, contour(domx, domx, (x,y)->pullback_func([x,y]), 
                title="pulled back targets $i"; kwargs...)
        )
    end
    return plot(plt_list_forward..., plt_list_pullback..., 
                plt_pulledback_reference..., layout=(3, length(sar.models)))
end


end
