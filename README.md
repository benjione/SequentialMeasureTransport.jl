# PSDModels
Simple implementation of models of the type
$$f(x) = \sum_{ij} k(x,x_i) B_{ij} k(x,x_j)$$
where $k$ is a Kernel function. This is bases on a paper by Marteau-Ferey et al. (see [1]).

Currently, the usage of this library is the following:

```julia
using PSDModels
using KernelFunctions # get your kernels from here
using LinearAlgebra

# positive function to be approximated from samples
f(x) = 2*(x-0.5)^2 * (x+0.5)^2

# Generate some data
N = 100
X = collect(range(-1, 1, length=N))
Y = f.(X)

# Create a model
k = MaternKernel(ν=1.5) # kernel to be used
model = PSDModel(X, Y, k)

# evaluate the model
model(0.2)

# modify the model
model = 2*model
```

### References
[1] U. Marteau-Ferey, F. Bach, and A. Rudi, “Non-parametric Models for Non-negative Functions” url: https://arxiv.org/abs/2007.03926
