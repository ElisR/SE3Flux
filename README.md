# SE3Flux.jl

Implementing $SE(3)$-equivariant [Tensor Field Networks](https://arxiv.org/abs/1802.08219)[^1] (TFN) with [`Flux.jl`](https://github.com/FluxML/Flux.jl).
This repository is not affiliated with the original paper in any way.
Rather, the official library implementation of this work is the PyTorch variant [e3nn](https://github.com/e3nn/e3nn) and the JAX variant [e3nn-jax](https://github.com/e3nn/e3nn-jax).
The Julia implementations of TFN and its derivatives are (as far as I know) non-existent.
This is a first step before I rewrite these equivariant layers in the style of [`GeometricFlux.jl`](https://github.com/FluxML/GeometricFlux.jl), which represents features with a graph.

For those unfamiliar with TFN, I motivate this architecture at the end of this README.

## Usage

A disadvantage of TFN is that one must keep track of which rotation representation any given feature vector belongs to.
As such, the format for storing feature vectors is (unavoidably) messier than a simple array.

### Example

Let's consider the architecture used for the shape classification example in the [Tensor Field Networks](https://github.com/tensorfieldnetworks/tensorfieldnetworks) paper, implemented in `/Shape_Classification.ipynb`.
Here, the aim is to classify a bunch of Tetris-like blocks, of which there are 8 distinct types.
The intrinsic rotational invariance of the network means that after being shown just one example of each block, the classifier can be equally confident in recognising the blocks even when they are arbitrarily orientated.
Below is a cartoon of the _invariance_ of the output of the entire pipeline with respect to rotation of the input, a special case of equivariance with a trivial identity representation.

<p align="center">
    <img src="https://user-images.githubusercontent.com/19764906/213000338-b66906d0-5adf-414d-b2b9-b1ff17ae0d02.svg" width="200">
</p>

A diagram of the network architecture is shown below.
We keep track of the rotation representations of the feature vectors with $\ell$ (see README Appendix or paper itself).
Applying the convolution is equivalent to taking the tensor product with a filter that has its own rotation representation $\ell_f$.
Augmenting a representation $\ell_i$ with a filter $\ell_f$ produces a sum of representations $\ell_o$ in the range $| \ell_i - \ell_f | \leq \ell_o \leq \ell_i + \ell_f$ (resembling a triangle inequality between vectors).
As detailed in the paper, special care is taken to ensure that the tensor product transforms appropriately under rotation, by weighting different terms in this sum by so-called [Clebsch-Gordan coefficients](https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients).
In the network below, we choose to discard any $\ell > 1$ terms.

<p align="center">
<img src="https://user-images.githubusercontent.com/19764906/212673994-37282db1-4695-434d-ba52-a4ed7d3cd15c.svg" width=700>
</p>

In this repository, this is implemented with the following code block.
`Chain` is the `Flux.jl` structure that holds sequential layers.
At all points we must specify the number of channels for each $\ell$.
The most important (and complicated) component is the $SE(3)$-equivariant convolution layer `E3ConvLayer`.
The remaining components are non-linear and self-interaction layers, used to scale feature vectors pointwise and mix channels, respectively, done in such a way as to preserve equivariance.
We take care that the number of output and input channels for every layer is correctly specified at the time of construction.
(Above, the number of channels with a particular representation is given by $[n]$.)
The final three steps are standard pooling, dense and softmax layers used for classification of the eight shapes.

```julia
centers = range(0f0, 3.5f0; length=4) |> collect

classifier = Chain(
                SIWrapper([1 => 4]),
                E3ConvLayer([4], [[(0, 0) => [0], (0, 1) => [1]]], centers),
                SIWrapper([4 => 4, 4 => 4]),
                NLWrapper([4, 4]),
                E3ConvLayer([4, 4], [[(0, 0) => [0], (0, 1) => [1]],
                                     [(1, 0) => [1], (1, 1) => [0, 1]]], centers),
                SIWrapper([8 => 4, 12 => 4]),
                NLWrapper([4, 4]),
                E3ConvLayer([4, 4], [[(0, 0) => [0]], [(1, 1) => [0]]], centers),
                SIWrapper([8 => 4]),
                NLWrapper([4]),
                PLayer(),
                Dense(4 => 8)
) |> gpu;
```

## Appendix

### The Need for Rotation Equivariant Networks

Rotational and translational symmetry are common in nature.
It is therefore useful to have neural networks that can exploit this simplified structure of many physical problems, without needing to carry around redundant information.
This is solved by making layers _equivariant_ (reviewed in detail in the [Geometric Deep Learning](https://arxiv.org/abs/2104.13478) textbook), meaning that the output transforms appropriately when the input is transformed.
There are many ways to design such neural network architectures, often with a trade-off between expressivity and computational cost.

The _Tensor Field Network_ (TFN) is built from matrix representations of rotations and acts on point clouds of features.
Translation symmetry is trivially upheld by only ever considering the relative displacement between points.
The advantage of TFN is that features can be complex physical quantities, but this expressivity comes with additional cost compared to some alternatives, especially because it essentially acts on an "all-to-all" graph.
Some follow-up works such as [SE(3)-Transformers](https://arxiv.org/abs/2006.10503) are more performant.

### Basis of Tensor Field Networks

A neural network acts on feature vectors, which are sometimes physical quantities.
These quantities can transform differently under rotation depending on their _rotation representation_, indexed by the non-negative integer $\ell$.
This distinction is obvious if we compare a scalar (such as an object's mass) to a vector (such as its velocity, where "vector" refers to a geometric quantity, and not just a one-dimensional array of numbers):
One does not change at all under rotation, while the other changes direction.
In representation theory (the study of maps from abstract symmetry groups to linear transformations, i.e., matrices), we say that the scalar transforms under the $\ell = 0$ irreducible representation (irrep), whereas the vector transforms under the $\ell = 1$ irrep.
(In quantum physics, these matrices act on the wavefunction of a particle, in which case $\ell$ corresponds to _angular momentum_.)
The matrices in an irreducible representation have dimension $(2\ell + 1) \times (2\ell + 1)$, so the transformation of quantities with higher $\ell$ are more costly to compute.

[^1]: Despite its name, this is completely unrelated "Tensor Networks" used in condensed matter, for which you instead want [`ITensors.jl`](https://github.com/ITensor/ITensors.jl) or a related package.