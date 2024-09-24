# # [1D KR SoS map from data](@id oned_kr_sos_map_data_example)
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
# Let us generate some data from this distribution
X = rand(mixture_pdf, 1, 5000);

# In order to fit the density, we need to define a loss function and minimize over it.
# `SequentialMeasureTransport.jl` provides divergences and losses as well as functions to minimize PSD models.
# These can be found in the `SequentialMeasureTransport.Statistics` module.
using SequentialMeasureTransport.Statistics


# We are using the log likelihood to fit the distribution to the data.
# This is basically the KL-divergece, which can be decomposed as

# ```math
# \begin{align}
# \mathcal{D}_{KL}(p||q) &= \int \log(\frac{p(x)}{q(x)}) p(x) dx - \int p(x) dx + \int q(x) dx \\
# &= \int p(x) (\log(p(x))-1) dx - \int \log(q(x)) p(x) dx + \int q(x) dx
# \end{align}
# ```
# where the second part can be estimated by samples from `p(x)`.
ML_fit!(model, eachcol(X), trace=false);

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