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

function plot_distances(sar::SelfReinforcedSampler,
            target_distribution::Function,
            L_domain, R_domain;
            N_samples=150,
            savefig_path=nothing,
            kwargs...
        )
    ranges = [range(l, r, length=N_samples) 
                    for (l,r) in zip(L_domain, R_domain)]
    iter = Iterators.product(ranges...)
    KL_div(a,b) = (1/length(a)) * sum(a .* log.(a ./ b))
    chi2_distance(a,b) = (1/length(a)) * sum((a .- b).^2 ./ b)
    hell_distance(a,b) = (1/length(a)) * sum((a.^0.5 .- b.^0.5).^2)
    
    tuple_to_vec(func) = x->func([x...])
    tar_vec = iter |> collect |> x->tuple_to_vec(target_distribution).(x)

    KL_list = []
    chi2_list = []
    hell_list = []
    for (i, model) in enumerate(sar.models)
        pdf_func = PSDModels.pushforward_pdf_function(sar;layers=collect(1:i))
        pdf_vec = iter |> collect |> x->tuple_to_vec(pdf_func).(x)

        KL = KL_div(tar_vec, pdf_vec)
        chi2 = chi2_distance(tar_vec, pdf_vec)
        hell = hell_distance(tar_vec, pdf_vec)

        push!(KL_list, KL)
        push!(chi2_list, chi2)
        push!(hell_list, hell)
    end
    N = length(sar.models)
    plt = plot(1:N, KL_list, label="KL"; kwargs...)
    plot!(plt, 1:N, chi2_list, label="\$ \\chi^2 \$"; kwargs...)
    plot!(plt, 1:N, hell_list, label="Hellinger"; kwargs...)
    xlabel!(plt, "Layers")
    ylabel!(plt, "Distance")

    if savefig_path !== nothing
        savefig(plt, savefig_path*"sampler_distances.pdf")
    end
    return plt
end

function plot_sampling_distribution2D(sar::SelfReinforcedSampler{d, T};
                                amount_samples=100,
                                give_sampling_vector=false) where {d, T<:Number}
    sample_list = Matrix{Vector{T}}(undef, length(sar.samplers), amount_samples)
    for i=1:length(sar.samplers)
        sample_list[i,:] = PSDModels.pushforward.(Ref(sar), PSDModels.sample_reference(sar, amount_samples); layers=collect(1:(i-1)))
    end
    plt_list = []
    for i=1:length(sar.samplers)
        plt = scatter([x[1] for x in sample_list[i,:]], [x[2] for x in sample_list[i,:]], label="Layer $i")
        push!(plt_list, plt)
    end
    if give_sampling_vector
        return plt_list, sample_list
    end
    return plt_list
end

end
