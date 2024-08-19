# # [1D KR SoS map from density](@id oned_kr_sos_map_example)
#
# Let's first load `SequentialMeasureTransport.jl` and other packages we use in this tutorial.
#

using SequentialMeasureTransport
import SequentialMeasureTransport as SMT

using Distributions
using Plots

#
# We are going to use a SoS Knothe-Rosenblatt map. To do so, we first
# create a PSDModel with a given basis. The library support `ApproxFun.jl` for the basis.
# Let us first load `ApproxFun.jl`.
#
using ApproxFun

#
# We can now create a PSDModel with a Legendre basis. Most often, we want to use the Legendre basis
# when working with polynomial PSDModels, since it is orthonormal to the Lebesgue measure.
# The `:downward_closed` argument specifies that the index set used is downward closed.
# For 1D models, this is irrelevant.
# The last argument specifies the number of basis functions to use.
model = PSDModel(Legendre(0.0..1.0), :downward_closed, 10)

#
# Next, we want to fit a PSD model to a denisty.
# We will use a simple 1D density, which is a mixture of two Gaussians.
# We will use the `MixtureModel` from the `Distributions.jl` package.
#
mixture_pdf = MixtureModel([Normal(0.3, 0.05), Normal(0.7, 0.05)], [0.5, 0.5])

#
# Let us evaluate this density at some points in order to fit it.
X = rand(5000)
Y = pdf.(Ref(mixture_pdf), X)
nothing

#
# In order to fit the density, we need to define a loss function and minimize over it.
# `SequentialMeasureTransport.jl` provides divergences and losses as well as functions to minimize PSD models.
# These can be found in the `SequentialMeasureTransport.Statistics` module.
using SequentialMeasureTransport.Statistics

#
# We will use the χ2 divergence between two densities here and fit our model to the available samples.
# First, we need to modify X, since these functions assume that X is an array of an array, which makes more sense in higher dimensions.
Chi2_fit!(model, [[x] for x in X], Y, trace=false)
nothing

#
# Now, we can use this model to create a SoS Knothe-Rosenblatt map.
smp = Sampler(model)

#
# We can now sample from the density using either the `sample` function or the `rand` interface.
S = rand(smp, 5000)
histogram(vcat(S...), normed=true, bins=60, label="Sampled")


#
# We can also evaluate the density at points using the `pdf` interface from `Distributions.jl`.
plot!(x -> pdf(mixture_pdf, x), 0.0, 1.0, label="True density", linewidth=2)
plot!(x -> pdf(smp, [x]), 0.0, 1.0, label="Estimated density", linewidth=2, linestyle=:dot)