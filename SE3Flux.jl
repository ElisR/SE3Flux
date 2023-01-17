module SE3Flux

using Flux
using MLUtils
using CUDA
using NNlib
using TensorCast
using ChainRulesCore

using Memoize
using Symbolics
using SphericalHarmonics

include("utils.jl")
include("Spherical.jl")
include("alt_parallel.jl")

export cart_to_sph
export pairwise_rs

include("TFNLayers.jl")

export SIWrapper
export E3ConvLayer
export NLWrapper
export PLayer

end