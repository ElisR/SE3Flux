# SE3Flux.jl

Implementing $SE(3)$-equivariant [Tensor Field Networks](https://arxiv.org/abs/1802.08219) with [`Flux.jl`](https://github.com/FluxML/Flux.jl).

### Motivation for Rotation Equivariant Networks

Rotational and translational symmetry are common in nature.
It is therefore useful to have neural networks that can exploit this simplified structure of many physical problems, without needing to carry around redundant information.

A neural network acts on feature vectors, which are sometimes physical quantities.
These quantities can transform differently under rotation depending on their _rotation representation_, indexed by the non-negative integer $\ell$.
This distinction is obvious when comparing a scalar quantity (such as an object's mass) to a vector quantity (such as its velocity):
One does not change at all under rotation, while the other changes direction.
In representation theory (the study of maps from abstract symmetry groups to linear transformations, i.e., matrices), we say that the scalar transforms under the $\ell = 0$ representation, whereas the vector transforms under the $\ell = 1$ representation.
The matrices in representations have dimension $(2\ell + 1) \times (2\ell + 1)$, so the transformation of quantities with higher $\ell$ are more costly to compute.


### Example

Let's consider the architecture used for the shape classification example in the [Tensor Field Networks](https://github.com/tensorfieldnetworks/tensorfieldnetworks) paper.
Here, the aim is to classify a bunch of Tetris-like blocks, of which there are 8 distinct types.
The intrinsic rotational invariance of the network means that after being shown just one example of each block, the classifier can be equally confident in recognising the blocks even when they are arbitrarily orientated.

Here's the network architecture we wish to implement.
In this diagram we keep track of the rotation representations of the feature vectors, denoted by $\ell$.
Applying the convolution is equivalent to taking the tensor product with a filter that has its own rotation representation $\ell_f$.
Augmenting a representation $\ell_i$ with a filter $\ell_f$ produces a sum of representations $\ell_o$ in the range $| \ell_i - \ell_f | \leq \ell_o \leq \ell_i + \ell_f$ (resembling a triangle inequality between vectors).
As detailed in the paper, special care is taken to ensure that the tensor product transforms appropriately under rotation, by weighting different terms in this sum by so-called [Clebsch-Gordan coefficients](https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients).
In the network below, choose to discard any $\ell > 1$ terms.


![TFN architecture for shape classification.](https://user-images.githubusercontent.com/19764906/212673994-37282db1-4695-434d-ba52-a4ed7d3cd15c.svg)

In this repository, this is implemented with the following code block.
`Chain` is the `Flux.jl` structure that holds sequential layers.
At all points we must specify the number of channels for each $\ell$.
The most important (and complicated) component is the $SE(3)$-equivariant convolution layer `E3ConvLayer`.
The remaining components are non-linear and self-interaction layers, used to scale feature vectors pointwise and mix channels, respectively, done in such a way as to preserve equivariance.
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
