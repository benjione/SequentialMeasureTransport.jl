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

# In practice, we do not have given access to this density, but instead, access
# to an unnormalized version of it. Let us define a function proportional to the density.
unnormlaized_density(x) = pdf(mixture_pdf, x)[1]

#
# Let us evaluate this density at some points from which we want to fit it.
X = rand(1, 5000)
Y = unnormlaized_density.(eachcol(X))
nothing

#
# In order to fit the density, we need to define a loss function and minimize over it.
# `SequentialMeasureTransport.jl` provides divergences and losses as well as functions to minimize PSD models.
# These can be found in the `SequentialMeasureTransport.Statistics` module.
using SequentialMeasureTransport.Statistics

# ## Fitting models using different divergences
# In order to fit our model to a given density, different statistical distances can be used.
# In the following, we are going to demonstrate different divergences implemented in the Statistics module.


# First, we consider the χ2 divergence.
model_chi2 = deepcopy(model)
Chi2_fit!(model_chi2, eachcol(X), Y, trace=false)
nothing

# Next, the KL-divergence.
model_KL = deepcopy(model)
KL_fit!(model_KL, eachcol(X), Y, trace=false)
nothing

#
# In order to get an approximation of the probability density, the approximated model has to be
# normalized.
# This can be done by using 
# ```julia
# normalize!(model)
# ```
# Alternatively, we can directly build a SoS Knothe-Rosenblatt map from it, which will automatically normalize it.
smp_chi2 = Sampler(model_chi2)
smp_KL = Sampler(model_KL)

#
# We can now sample from the density using either the `sample` function or the `rand` interface.
# We are only going to sample from the KL approximation in this example.
S = rand(smp_KL, 5000)
histogram(vcat(S...), normed=true, bins=60, label="Sampled")


#
# We can also evaluate the density at points using the `pdf` interface from `Distributions.jl`.
plot!(x -> pdf(mixture_pdf, x), 0.0, 1.0, label="True density", linewidth=2)
plot!(x -> pdf(smp_chi2, [x]), 0.0, 1.0, label="Estimated density χ2", linewidth=2, linestyle=:dot)
plot!(x -> pdf(smp_KL, [x]), 0.0, 1.0, label="Estimated density KL", linewidth=2, linestyle=:dot)