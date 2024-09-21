# SequentialMeasureTransport.jl

Sequential measure transport is a library to facilitate the construction of composed transport maps for the task of denisty estimation.

It comes shipped with triangular Knothe-Rosenblatt maps build from PSDModels (also known as Sum-of-Squares) but allows you to integrate your transport map of choice.
It comes with functionalities to create compositions of those, $\mathcal T_{\ell} = \mathcal Q_1 \circ \mathcal Q_2 \circ \dots \circ \mathcal Q_{\ell}$, so that each composition represents intermediate densities $\pi^{(\ell)}$,
```math
    \left(\mathcal T_{\ell}\right)_\sharp \rho = \pi^{(\ell)}
```

For a more detailed introduction of transport maps, see .

## Related packages

- [MParT](https://measuretransport.github.io/MParT/index.html), a package for triangular parameterization of transport maps.
- [DIRT](https://github.com/DeepTransport/deep-tensor), a matlab package for deep inverse Rosenblatt transport, also see [here](https://arxiv.org/abs/2106.04170)