using Flux
using CUDA
using NNlib
using TensorCast
using MLUtils

# SH and CG coefficients
using Symbolics
using SphericalHarmonics
include("Spherical.jl")
include("utils.jl")
include("alt_parallel.jl")

"""
One simple ℝ ≥ 0 -> ℝ function broadcasted across every elements of the array.
Function is a linear combination of basis functions ∑ᵢ aᵢ rbfᵢ(r), with learned weightings aᵢ.
"""
struct RLayer{A, V, Γ}
    as::A # TODO Maybe add another NN to copy TFN
    
    centers::V
    spacing::Γ
end

function RLayer(centers; init=Flux.glorot_uniform)
    n_basis = length(centers)

    spacing = (centers[end] - centers[1]) / n_basis
    RLayer(init(n_basis), centers, spacing)
end

function (R::RLayer)(radials)
    based = Flux.batch([@.(exp(- R.spacing * (radials - c)^2)) for c in R.centers])

    @reduce R_out[b, a, γ] := sum(k) R.as[k] * based[b, a, γ, k] 
end

Flux.@functor RLayer (as,)

struct FLayer{r, y}
    R::r # Radial NN

    Ys::y # SH functions for this ℓf
    ℓf::Int # Filter angular momentum # TODO Consider removing
end

function FLayer(Ys, centers::Vector{Float32})
    # Will later allow for custom spec
    R = RLayer(centers)

    ℓf = (length(Ys) - 1) ÷ 2
    FLayer(R, Ys, ℓf)
end

function (F::FLayer)(rr)
    # Apply R to the input radii
    rr_rs = @view rr[:, :, 1, :]
    R_out = F.R(rr_rs)

    # Multiply by SH components
    θs = @view rr[:,:,2,:]
    ϕs = @view rr[:,:,3,:]
    Y_out = Flux.batch([Y.(θs, ϕs) for Y in F.Ys])
    
    R_out .* Y_out
end

Flux.@functor FLayer (R,)

"""
Structure of convolution layer defined by (ℓi, ℓf) => [ℓo1, ℓo2, ...], where ℓi and ℓf are input and filter angular momentum representations.
Trainable component is NN `F` contains learnable radial basis functions and spherical harmonic transformations.
Non-trainable elements include precomputed CG coefficients.
"""
struct CLayer
    F::FLayer # Trainable NN

    CG_mats::Dict{Tuple{Int, Int}, CuArray{Float32}} # Dictionary of matrices, keyed by (ℓo, mo)
    ℓi::Int # Input ℓ
    ℓf::Int # Filter ℓ
    ℓos::Vector{Int}

    outkeys::Dict{Int, Int}
    ℓ_max::Int
end


function CLayer(((ℓi, ℓf), ℓos)::Pair{Tuple{Int, Int}, Vector{Int}}, centers::Vector{Float32}; ℓ_max::Int = 0)
    @assert ℓos ⊆ abs(ℓi - ℓf):(ℓi + ℓf) "Output `ℓo` not compatible with filter `ℓf` and input `ℓi`."

    Ys = generate_Yℓms(ℓf)
    F_NN = FLayer(Ys, centers)

    # Not necessarily choosing every possible output
    CG_mats::Dict{Tuple{Int, Int}, CuArray{Float32}} = Dict()
    for ℓo in ℓos
        CG_tensor = generate_CG_matrices(ℓi, ℓf, ℓo)
        for (i, mo) in enumerate(-ℓo:ℓo)
            CG_mats[(ℓo, mo)] = cu(@view CG_tensor[:, :, i])
        end
    end

    # Storing where each element will go in the tuple
    # Needed because there will generically be gaps for non-existent ℓo in output.
    outkeys = Dict(ℓo => i for (i, ℓo) in enumerate(ℓos))
    
    CLayer(F_NN, CG_mats, ℓi, ℓf, ℓos, outkeys, max(ℓ_max, ℓos[end]))
end

"""
Forward pass of CLayer.
V indexed by [mi, b].

Returns a tuple of vectors.
"""
function (C::CLayer)(rr, V)
    F_out = C.F(rr)
    
    # Using Einstein summation convention for brevity, with similar speed
    # TODO Eventually investigate batched multiplication for all these
    @reduce F_tilde[mi, mf, a, γ] := sum(b) V[b, γ, mi] * F_out[b, a, γ, mf]

    L_tildes = [Flux.batch([C.CG_mats[(ℓo, mo)] .* F_tilde for mo in -ℓo:ℓo]) for ℓo in C.ℓos]
    Ls = [@reduce L[a, γ, mo] := sum(mi, mf) L_tilde[mi, mf, a, γ, mo] for L_tilde in L_tildes]
    Tuple((ℓo_pot ∈ C.ℓos) ? [Ls[C.outkeys[ℓo_pot]]] : Vector{eltype(Ls)}(undef, 0) for ℓo_pot in 0:C.ℓ_max)
end

function (C::CLayer)(rrV::Tuple)
    C(rrV...)
end

Flux.@functor CLayer (F,)

struct NLLayer{b, S}
    bias::b

    σ::S
end

function NLLayer(channels::Int; σ = Flux.elu, init = Flux.glorot_uniform)
    NLLayer(init(channels), σ)
end

"""
Take in a vector `Vs` of features `V`.
Each `V` indexed as `[a, γ, mo]` for point `a`, sample `γ` and irrep index `mo`.
"""
function (nl::NLLayer)(Vs)
    bVs = batch(Vs)
    norm = .√sum(bVs.^2, dims=3)
    bias_reshaped = reshape(nl.bias, 1, 1, 1, :)
    unbatch(@. nl.σ(norm + bias_reshaped) * bVs)
end

Flux.@functor NLLayer (bias,)

struct SILayer
    W
end

"""
Self-interaction layer specified in TFN.
"""
function SILayer((C_in, C_out)::Pair{Int, Int}; init=Flux.glorot_uniform)
    SILayer(init(C_out, C_in))
end

"""
Mixing a load of channels.
"""
function (si::SILayer)(Vs)
    Vs_batch = batch(Vs)
    @reduce Vs_mix[a, γ, mo, c_out] := sum(c_in) si.W[c_out, c_in] * Vs_batch[a, γ, mo, c_in]
    unbatch(Vs_mix)
end

Flux.@functor SILayer (W,)

struct PLayer end

"""
Pooling layer.
Assumes an input shaped `[a, γ, m]` for point `a`, sample `γ` and orbital `m`.
Also assume `ℓ = 0` since pooling happens for final feature final feature.
Output a matrix indexed by `[c, γ]`
"""
(pl::PLayer)(rssVs::Tuple) = pl(rssVs...)
function (pl::PLayer)(rss, (Vs,))
    #[dropdims(sum(V, dims=1); dims=1) for V in Vs]
    Vstack = batch(Vs)
    dropdims(sum(Vstack, dims=1); dims=(1, 3))'
end

"""
Structure for orchestrating all individual convolutions.
`Cs` holds all of the learnable convolution layers.
`n_cs` is the number of channels.
"""
struct E3ConvLayer{C, P}
    Cs::C

    n_cs::Vector{Int}
    pairings::P
end

"""
Constructor for plethora of point convolutions.
`n_cs` is number of channels for each representation, explicitly including any n_c=0.
`pairingss` takes form
```
[[(ℓi=0, ℓf01) => [ℓo011, ℓo012], ℓf02 => [ℓo021]],
 []
 [(ℓi=2, ℓf21) => [ℓo211]]]
```
where each pair is a valid CLayer pairing mapping filters to output angular momenta.
Each `ℓos` list assumed to be ordered in increasing order.

e.g. for Tetris example in TFN paper, we have:
E3ConvLayer([1], [[(0, 0) => [0], (0, 1) => [1]]]) # Layer 1
E3ConvLayer([4, 4], [[(0, 0) => [0], (0, 1) => [1]], [(1, 0) => [1], (1, 1) => [0, 1]]) # Layer 2
"""
function E3ConvLayer(n_cs::Vector{Int}, pairingss::T, centers::Vector{Float32}) where T
    @assert length(n_cs) == length(pairingss) "Filters must be specified for each rotation representation. Put empty `ℓfs` for any `n_c = 0`."

    ℓis_implied = 0:(length(n_cs) - 1)

    # Find maximal output dimension possible
    ℓ_max = 0
    for pairings in pairingss
        for (_, ℓos) in pairings
            ℓ_max = max(ℓ_max, ℓos[end])
        end
    end

    overall = []
    for (ℓi_implied, n_c, pairings) in zip(ℓis_implied, n_cs, pairingss)
        # Apply all filters
        in_channel = []
        for ((ℓi, ℓf), ℓos) in pairings
            @assert ℓi_implied == ℓi "An input ℓi is not consistent with its placement."

            channel_convs = [CLayer((ℓi, ℓf) => ℓos, centers; ℓ_max = ℓ_max) for _ in 1:n_c]

            # Single element things are messing it up
            p = ParallelPassenger(tuple_connect, Tuple(channel_convs))
            push!(in_channel, p)
        end

        # Assuming that all present ℓ has to be convolved
        # p_ℓi could be empty if there were no channels
        p_ℓi = (n_c > 0 && !isempty(in_channel)) ? ParallelPassenger(tuple_connect, Tuple(in_channel); singleton=true) : (rs, empty) -> Tuple(empty for _ in 0:ℓ_max)
        push!(overall, p_ℓi)
    end
    Cs = ParallelPassenger(tuple_connect, Tuple(overall))

    E3ConvLayer(Cs, n_cs, pairingss)
end

(e3::E3ConvLayer)(rssVs::Tuple) = e3(rssVs...)
(e3::E3ConvLayer)(rss, Vs) = (rss, e3.Cs(rss, Vs))

Flux.@functor E3ConvLayer (Cs,)