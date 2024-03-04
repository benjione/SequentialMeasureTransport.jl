# SequentialTransportMaps.jl

Code of the paper [Seqential Transport maps using SoS densities and $\alpha$-divergences](https://arxiv.org/abs/2402.17943)

Documentation comming soon.

This package allows for implementing sequential transport maps in order to estimate some probability distribution $\pi$, as first described in [Cui et al 2023]().
We consider both estimating $\pi$ from samples or acces to its unnormalized density.
In measure transport (a concept also known as normalizing flows in the deep learning community), $\pi$ is estimated using a transport map $\mathcal T$ (a diffeomorphic function), so that
$$
    \mathcal T_\sharp \mu(x) := \mu(\mathcal T^{-1}(x)) \det \nabla \mathcal T^{-1}(x) = \pi(x).
$$
Such transport maps gained popularity for generative modeling, since
$$
    \mathcal T(x) = y \qquad \text{with } x \sim \mu, y \sim \pi,
$$
making it easy to sample from $\pi$ if $\mu$ is a tractable distribution.
Since approximating $\pi$ could be too difficult, it has been proposed to use a sequence of intermediate bridging densities of increasing complexity, which is estimated sequentially.
Let this sequence be