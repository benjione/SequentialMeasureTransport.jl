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
            single_plot=true,
            titles=true,
            savefig_path=nothing,
            kwargs...
        )
    @assert length(sar.models) == length(target_list)
    domain = domain === nothing ? (-1.0, 1.0) : domain
    domx = range(domain[1], domain[2], length=N_plot)
    plt_list_forward = []
    for (i, model) in enumerate(sar.models)
        pdf_func = PSDModels.pushforward_pdf_function(sar;layers=collect(1:i))
        plt = contour(domx, domx, (x,y)->pdf_func([x,y]); kwargs...)
        if titles
            title!(plt, "\$ \\left( T_{\\leq $i} \\right)_{\\#} \\rho_{ref} \$")
        end
        push!(plt_list_forward, plt)
    end
    plt_list_pullback = []
    for (i, tar) in enumerate(target_list)
        pullback_func = PSDModels.pullback_pdf_function(sar, tar;layers=collect(1:i-1))
        if i==1
            plt = contour(domx, domx, (x,y)->pullback_func([x,y]); kwargs...)
            if titles
                title!(plt, "\$ \\pi_1 \$")
            end
            push!(plt_list_pullback, plt)
        else
            plt = contour(domx, domx, (x,y)->pullback_func([x,y]);  kwargs...)
            if titles
                title!(plt, "\$T_{ \\leq $(i-1) }^\\# \\pi_{$i } \$")
            end
            push!(plt_list_pullback, plt)
        end
    end
    # plt_pulledback_reference = []
    # for (i, tar) in enumerate(target_list)
    #     pullback_func = PSDModels.pullback_pdf_function(sar, tar;layers=collect(1:i))
    #     push!(plt_pulledback_reference, contour(domx, domx, (x,y)->pullback_func([x,y]), 
    #             title="\$T_{ \\leq $(i) }^\\# \\pi_{$i } "; kwargs...)
    #     )
    # end
    if single_plot
        plt = plot(plt_list_forward..., plt_list_pullback..., 
                    layout=(2, length(sar.models)))
        if savefig_path !== nothing
            savefig(plt, savefig_path*"sampler_overview.pdf")
        end
        return plt
    else
        if savefig_path !== nothing
            for (i, plt) in enumerate(plt_list_forward)
                savefig(plt, savefig_path*"sampler_forward_$i.pdf")
            end
            for (i, plt) in enumerate(plt_list_pullback)
                savefig(plt, savefig_path*"sampler_pullback_targets_$i.pdf")
            end
        end
        return plt_list_forward, plt_list_pullback
    end
end


end
