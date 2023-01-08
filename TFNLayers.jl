using Flux
using CUDA
using NNlib
using TensorCast

# SH and CG coefficients
using Symbolics
using SphericalHarmonics
include("Spherical.jl")
include("utils.jl")

"""
One simple ℝ ≥ 0 -> ℝ function broadcasted across every elements of the array.
Function is a linear combination of basis functions ∑ᵢ aᵢ rbfᵢ(r), with learned weightings aᵢ.
"""
struct RLayer{A, V, Γ}
    as::A
    # TODO Maybe add another NN to copy TFN
    
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

struct FLayer
    # Radial NN
    R::RLayer

    Ys::Vector{Function} # SH functions for this ℓf
    ℓf::Int # Filter angular momentum # TODO Consider removing
end

function FLayer(Ys::Vector{Function}, centers::Vector{Float32})
    # Will later allow for custom spec
    R = RLayer(centers)

    ℓf = (length(Ys) - 1) ÷ 2
    FLayer(R, Ys, ℓf)
end

# Dimension needs to be made one bigger
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

struct CLayer
    F::FLayer # Trainable NN

    CG_mats::Dict{Tuple{Int, Int}, CuArray{Float32}} # Dictionary of matrices, keyed by (ℓo, mo)
    ℓi::Int # Input ℓ
    ℓf::Int # Filter ℓ
    ℓos::Vector{Int}
    ℓms::Vector{Tuple{Int, Int}} # Specifying output order
end

# Constructor
function CLayer(((ℓi, ℓf), ℓos)::Pair{Tuple{Int, Int}, Vector{Int}}, centers::Vector{Float32})
    @assert ℓos ⊆ abs(ℓi - ℓf):(ℓi + ℓf) "Output `ℓo` not compatible with filter `ℓf` and input `ℓi`."

    Ys = generate_Yℓms(ℓf)
    F_NN = FLayer(Ys, centers)

    # Not necessarily choosing every possible output
    ℓms::Vector{Tuple{Int, Int}} = []
    CG_mats::Dict{Tuple{Int, Int}, CuArray{Float32}} = Dict()
    for ℓo in ℓos
        for mo in -ℓo:ℓo
            push!(ℓms, (ℓo, mo))
            CG_mat = zeros(Float32, (2ℓi + 1, 2ℓf + 1))
            for (i_i, mi) in enumerate(-ℓi:ℓi)
                for (i_f, mf) in enumerate(-ℓf:ℓf)
                    CG_mat[i_i, i_f] = cg(ℓi, mi, ℓf, mf, ℓo, mo)
                end
            end
            CG_mats[(ℓo, mo)] = cu(CG_mat)
        end
    end
    
    CLayer(F_NN, CG_mats, ℓi, ℓf, ℓos, ℓms)
end

"""
Forward pass of CLayer
V indexed by [mi, b]
"""
function (C::CLayer)(rr, V)
    F_out = C.F(rr)
    
    # Using Einstein summation convention for brevity
    # Speed seems comparable
    # TODO Eventually investigate batched multiplication for all these
    @reduce F_tilde[mi, mf, a, γ] := sum(b) V[b, γ, mi] * F_out[b, a, γ, mf]

    L_tildes = Tuple(Flux.batch([C.CG_mats[(ℓo, mo)] .* F_tilde for mo in -ℓo:ℓo]) for ℓo in C.ℓos)
    Tuple(@reduce L[a, γ, mo] := sum(mi, mf) L_tilde[mi, mf, a, γ, mo] for L_tilde in L_tildes)
    #[dropdims(sum(L_tilde, dims=(1,2)), dims=(1,2)) for L_tilde in L_tildes]
end

Flux.@functor CLayer (F,)

#=
"""
    E3ConvLayer(S::Unsigned, n_ℓs::Vector{Unsigned}, filter_ℓ::Unsigned)

Constructor for `S` points, `[n_ℓ]` channels for each `ℓ`, `filter_ℓ` is angular momentum of filter.

`n_ℓs` is a vector `[n_0, n_1, ..., n_L]` where `ℓ=L` is the largest nonzero angular momentum feature.
(Preceding `n_i=0` entries must be present.)
Spends time generating dictionary of SH functions.
"""
function E3ConvLayer(S::Unsigned, n_ℓs::Vector{Unsigned}, ℓ_filter::Unsigned)
=#