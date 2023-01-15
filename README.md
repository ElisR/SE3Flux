# SE3Flux.jl

Implementing $SE(3)$-equivariant neural networks with `Flux.jl`.

### Example

Let's consider the architecture used for the shape classification example in the [Tensor Field Networks](https://github.com/tensorfieldnetworks/tensorfieldnetworks) paper.
Here, the aim is to classify a bunch of Tetris-like blocks, of which there are 8 distinct types.
The intrinsic rotational symmetry of the network means that even after being shown one example of each block, the classifier can be equally confident in recognising blocks that have been arbitrarily rotated.

Here's the network architecture, and here's how it's implemented in this repository.
`Chain` is the `Flux.jl` structure that holds sequential layers.
The most important (and complicated) component is the $SE(3)$-equivariant convolution layer `E3ConvLayer`.

```julia
centers = range(0f0, 3.5f0; length=4) |> collect

classifier = Chain(
                SIWrapper([1 => 4]),
                E3ConvLayer([4], [[(0, 0) => [0], (0, 1) => [1]]], centers),
                SIWrapper([4 => 4, 4 => 4]),
                NLWrapper([4, 4]),
                E3ConvLayer([4, 4], [[(0, 0) => [0], (0, 1) => [1]], [(1, 0) => [1], (1, 1) => [0, 1]]], centers),
                SIWrapper([8 => 4, 12 => 4]),
                NLWrapper([4, 4]),
                E3ConvLayer([4, 4], [[(0, 0) => [0]], [(1, 1) => [0]]], centers),
                SIWrapper([8 => 4]),
                NLWrapper([4]),
                PLayer(),
                Dense(4 => 8)
) |> gpu;
```