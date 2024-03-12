# SequentialMeasureTransport.jl

Code of the paper [Seqential Transport maps using SoS densities and $\alpha$-divergences](https://arxiv.org/abs/2402.17943)

A more detailed documentation is comming soon.

This package allows for implementing sequential transport maps in order to estimate some probability distribution $`\pi`$, as first described in [Cui et al 2023](https://arxiv.org/abs/2106.04170).
We consider both estimating $`\pi`$ from samples or acces to its unnormalized density.
In measure transport (a concept also known as normalizing flows in the deep learning community), $`\pi`$ is estimated using a transport map $`\mathcal T`$ (a diffeomorphic function), so that
```math
    \mathcal T_\sharp \mu(x) := \mu(\mathcal T^{-1}(x)) \det \nabla \mathcal T^{-1}(x) = \pi(x).
```
Such transport maps gained popularity for generative modeling, since
```math
    \mathcal T(x) = y \qquad \text{with } x \sim \mu, y \sim \pi,
```
making it easy to sample from $`\pi`$ if $`\mu`$ is a tractable distribution.
Since approximating $`\pi`$ could be too difficult, it has been proposed to use a sequence of intermediate bridging densities $`\{\pi^{(\ell)}\}_{\ell = 1}^{L}`$ of increasing complexity, which is estimated sequentially.
Sequential transport builds maps $`\mathcal T_{\ell} = \mathcal Q_1 \circ \dots \circ \mathcal Q_{\ell}`$ so that
```math
    \left(\mathcal T_{\ell}\right)_\sharp \mu = \widetilde{\pi}^{(\ell)} \approx \pi^{(\ell)}
```
and so that each map $`\mathcal Q_{\ell}`$ is learned by using
```math
    \left(\mathcal Q_{\ell}\right)_\sharp \mu \approx \mathcal T_{\ell -1}^\sharp \pi^{(\ell)}.
```

For know, we only support Knothe-Rosenblatt maps for $`\mathcal Q`$ but plan to allow for also using normalizing flows and be able to mix both.
