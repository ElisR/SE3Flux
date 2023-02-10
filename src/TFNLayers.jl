"""
Structure for learned radial function, which is unique to every channel.
Holds learned weight vector `as` ``\{a_i\}`` and fixed `spacing` ``\\rho`` and vector `centers` ``\{c_i\}``.
"""
struct RLayer{A, V, Γ}
    as::A
    
    centers::V
    spacing::Γ
end

"""
    `RLayer(centers; init = Flux.glorot_uniform)`

Constructor for `RLayer`, which holds a linear combination of radial basis functions ``\\sum_i a_i f_i(r)``, with learned weightings ``a_i`` initialised according to `init` function.
Radial basis functions by default chosen to be ``f_i(r) = e^{- \\rho (r - c_i)^2}``.
Set ``\{c_i\}`` fixed by `centers` in constructor, assumed to be in increasing order, while `\\gamma` is fixed to mean spacing between ``\{c_i\}``.
"""
function RLayer(centers; init=Flux.glorot_uniform)
    n_basis = length(centers)
    spacing = (centers[end] - centers[1]) / n_basis
    RLayer(init(n_basis), centers, spacing)
end

"""
    `rlayer(rrs)`

Forward pass of an ``\\mathbb{R} \\geq 0 \\mapsto \\mathbb{R}`` function broadcasted across every element of an array of pairwise radii `rrs`.
`rrs` consists of radial distances between all pairs of points, stored in `[b, a, B]` format, where `b` and `a` index the distance ``r_{ab}`` between the particles ``b`` and ``a`` respectively, where `B` is the batch index.
"""
function (R::RLayer)(radials)
    # TODO Allow for arbitrary radial basis functions in constructor.
    based = Flux.batch([@.(exp(- R.spacing * (radials - c)^2)) for c in R.centers])
    @reduce R_out[b, a, γ] := sum(k) R.as[k] * based[b, a, γ, k]
end

Flux.@functor RLayer (as,)

"""
Structure for generating filters from TFN.
`R` is the radial neural network.
`Ys` is a list of **real** spherical ``Y_{\\ell m}`` harmonics, ordered as ``[-\\ell_f, -(\\ell_f-1), \\ldots, \\ell_f-1, \\ell_f]``.
"""
struct FLayer{r, y}
    R::r

    Ys::y
    ℓf::Int
    tol::Real
end

"""
    `FLayer(Ys, centers; tol = 1f-7)`

Constructor for `FLayer`, which constructs a filter with fixed rotation representation.
"""
function FLayer(Ys, centers; tol = 1f-7)
    R = RLayer(centers)

    ℓf = (length(Ys) - 1) ÷ 2
    FLayer(R, Ys, ℓf, tol)
end

function (F::FLayer)(rr)
    # Apply R to the input radii
    rr_rs = @view rr[:, :, 1, :]
    R_out = F.R(rr_rs)

    # Mask out cases where `r_ab = 0`
    # Needed since angle is undefined and hence not equivariant
    mask = @. abs(rr_rs) > tol

    # Calculate SH components
    θs = @view rr[:,:,2,:]
    ϕs = @view rr[:,:,3,:]
    Y_out = Flux.batch([Y.(θs, ϕs) for Y in F.Ys])
    
    R_out .* Y_out .* mask
end

Flux.@functor FLayer (R,)

"""
Structure of convolution layer defined by (ℓi, ℓf) => [ℓo1, ℓo2, ...], where ℓi and ℓf are input and filter angular momentum representations.
Trainable component is NN `F` contains learnable radial basis functions and spherical harmonic transformations.
Non-trainable elements include precomputed CG coefficients.
"""
struct CLayer
    F::FLayer # Trainable NN

    CG_mats # Dictionary of matrices, keyed by (ℓo, mo)
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
    CG_mats = Dict()
    for ℓo in ℓos
        CG_tensor = generate_CG_matrices(ℓi, ℓf, ℓo)
        for (i, mo) in enumerate(-ℓo:ℓo)
            CG_mats[(ℓo, mo)] = (@view CG_tensor[:, :, i]) |> gpu
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

"""
    NLLayer(channels; σ = Flux.elu, init = Flux.glorot_uniform)

Nonlinear layer specified in TFN.
Acts as scalar transformation in each `ℓ` space.
Nonlinear element given by `σ` defines ``\\sigma^{\\ell} : \\mathbb{R} \\mapsto \\mathbb{R}``.
Bias vector `bias` initialised according to `init`.
"""
function NLLayer(channels::Int; σ = Flux.elu, init = Flux.glorot_uniform)
    NLLayer(init(channels), σ)
end

"""
    nl(Vs)

Forward pass of `NLLayer` Take in a vector `Vs` of features `V`.
Each `V` indexed as `[a, γ, m]` for point `a`, sample `γ` and irrep index `m`.
Outputs quantities from TFN paper:

``\\eta^{(l)}\\left(\\|V\\|_{a c}^{(l)}+b_c^{(l)}\\right) V_{a c m}^{(l)}``

where

``\\|V\\|_{a c}^{(l)}:=\\sqrt{\\sum_m\\left|V_{a c m}^{(l)}\\right|^2}``
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
    SILayer(C_in => C_out; init=Flux.glorot_uniform)

Self-interaction layer specified in TFN, used to scale feature vectors elementwise.
Mixes components of feature vectors with the same representation at each point.
`C_in` and `C_out` determine the number of input and output channels.
`init` determines the initialisation of the weight matrix ``W``.

"""
function SILayer((C_in, C_out)::Pair{Int, Int}; init=Flux.glorot_uniform)
    SILayer(init(C_out, C_in))
end

"""
    si(Vs)

Performing the self-interaction according to
``\\sum_{c'}^{C_{\\text{in}}} W_{c c'}^{(\\ell)} V_{ac'm}^{(\\ell)}``

`Vs` is a vector of `V`s (one for each channel), where each `V` is indexed differently to display equation.
Indexing follows `V[a, γ, m]`, where `γ` is the batch index.
"""
function (si::SILayer)(Vs)
    Vs_batch = batch(Vs)
    @reduce Vs_mix[a, γ, m, c_out] := sum(c_in) si.W[c_out, c_in] * Vs_batch[a, γ, m, c_in]
    unbatch(Vs_mix)
end

Flux.@functor SILayer (W,)

struct PLayer end

"""
Pooling layer.
Assumes an input shaped `[a, γ, m]` for point `a`, sample `γ` and orbital `m`.
Also assume `ℓ = 0` since pooling happens for final feature final feature.
Output a matrix indexed by `[c, γ]`.
"""
(pl::PLayer)(rssVs::Tuple) = pl(rssVs...)
function (pl::PLayer)(rss, (Vs,))
    Vstack = batch(Vs)
    dropdims(sum(Vstack, dims=1); dims=(1, 3))' # TODO Turn this into NNlib.meanpool call
end

"""
Structure for orchestrating all individual convolutions.
`Cs` holds all of the learnable convolution layers.
`n_cs` is the number of channels.
"""
struct E3ConvLayer{C, P}
    Cs::C

    n_cs::Vector{Int}
    pairingss::P
end

"""
Constructor for plethora of point convolutions.
`n_cs` is number of channels for each representation, explicitly including any n_c=0.
`pairingss` takes form
```jldoctest
[[(ℓi=0, ℓf01) => [ℓo011, ℓo012], ℓf02 => [ℓo021]],
 []
 [(ℓi=2, ℓf21) => [ℓo211]]]
```
where each pair is a valid CLayer pairing mapping filters to output angular momenta.
Each `ℓos` list assumed to be ordered in increasing order.

e.g. for Tetris example in TFN paper, we have:
```jldoctest
conv1 = E3ConvLayer([1], [[(0, 0) => [0], (0, 1) => [1]]])
conv2 = E3ConvLayer([4, 4], [[(0, 0) => [0], (0, 1) => [1]], [(1, 0) => [1], (1, 1) => [0, 1]])
```
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

struct SIWrapper
    p
end

"""
Constructor for self-interaction.
Used to mix features with a particular rotation representation and increase/decrease the number of channels.
`pairs::Vector{Pair{Int, Int}}` takes the form `[ni0 => no0, ni1 => no1, ni2 => no2, ...]``.
Each pair denotes (in order) the number of input and output channels in each representation, starting from `ℓ = 0`.
Forward pass calls `SILayer` on each rotation representation.

e.g. for Tetris example in TFN paper, we have:
```jldoctest
si2 = SIWrapper([4 => 4, 4 => 4])
```
"""
function SIWrapper(pairs::Vector{Pair{Int, Int}})
    SIs = Tuple(SILayer(in => out) for (in, out) in pairs)
    p = Passenger(triv_connect, SIs...)

    SIWrapper(p)
end

(swir::SIWrapper)(rxs...) = swir.p(rxs...)

Flux.@functor SIWrapper

struct NLWrapper
    p
end

"""
Constructor for non-linearity.
Used to apply point-wise non-linearity to each rotation representationsuch that equivariance is preserved.
`channels::Vector{Int}` takes the form `[n0, n1, n2, ...]``.
Each `nℓ` in the ordered vector is the number of channels in representation `ℓ``, starting from `ℓ = 0`.
Forward pass calls `NLLayer` on each rotation representation.

e.g. for Tetris example in TFN paper, we have:
```jldoctest
nl1 = SIWrapper([4, 4])
```
"""
function NLWrapper(channels::Vector{Int})
    NLs = Tuple(NLLayer(channel) for channel in channels)
    p = Passenger(triv_connect, NLs...)

    NLWrapper(p)
end

(nlr::NLWrapper)(rxs...) = nlr.p(rxs...)

Flux.@functor NLWrapper