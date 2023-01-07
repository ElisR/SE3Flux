using Flux

# SH and CG coefficients
using Symbolics
using SphericalHarmonics
include("Spherical.jl")

# -------------------
# R layer for basis functions
# -------------------

"""
One simple ℝ ≥ 0 -> ℝ function broadcasted across every elements of the array.
Function is a linear combination of basis functions ∑ᵢ aᵢ rbfᵢ(r), with learned weightings aᵢ.
"""
struct RLayer
    as::Vector{Float32}
    # TODO Maybe add another NN to copy TFN
    
    centers::Vector{Float32}
    γ::Float32
end

function RLayer(centers; init=Flux.glorot_uniform)
    n_basis = length(centers)
    as = init(n_basis)

    γ = (centers[end] - centers[1]) / n_basis
    RLayer(as, centers, γ)
end

function (R::RLayer)(radials)
    reduce(+, [a * @.(exp(- R.γ * (radials - c)^2)) for (a, c) in zip(R.as, R.centers)])
end

Flux.@functor RLayer (as,)

# -------------------
# F layer which sums
# -------------------

struct FLayer
    #R::Chain # Radial NN
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
    rr_radials = rr[:, :, 1, :] # TODO Possibly change this back to view

    R_out = F.R(rr_radials)

    # Multiply by SH components
    θs = @view rr[:,:,2,:]
    ϕs = @view rr[:,:,3,:]
    Y_out = Flux.batch([Y.(θs, ϕs) for Y in F.Ys])
    
    R_out .* Y_out
end

Flux.@functor FLayer (R,)

# -------------------
# Convolution layer
# -------------------

struct CLayer
    F::FLayer # Trainable NN
    CG_mats::Dict{Tuple{Int, Int}, CuArray{Float32}} # Dictionary of matrices, keyed by (ℓo, mo)

    ℓi::Int # Input ℓ
    ℓf::Int # Filter ℓ
    ℓms::Vector{Tuple{Int, Int}} # Specifying output order
end

# Constructor
function CLayer(ℓi::Int, ℓf::Int, ℓos::Vector{Int}, centers::Vector{Float32})
    @assert ℓos ⊆ abs(ℓi - ℓf):(ℓi + ℓf) "Output `ℓo` not compatible with filter `ℓf` and input `ℓi`."

    Ys = generate_Yℓms(ℓf)
    F_NN = FLayer(Ys, centers)

    # Not going to choose every one
    ℓms::Vector{Tuple{Int, Int}} = []
    CG_mats::Dict{Tuple{Int, Int}, CuArray{Float32}} = Dict()
    for ℓo in ℓos
        for mo in -ℓo:ℓo
            push!(ℓms, (ℓo, mo))
            CG_mat = zeros(Float32, (2ℓi + 1, 2ℓf + 1))
            for (i_i, mi) in enumerate(-ℓi:ℓi)
                for (i_f, mf) in enumerate(-ℓf:ℓf)
                    # TODO Check that ordering of f and i is correct
                    # Currently giving zero
                    CG_mat[i_i, i_f] = cg(ℓi, mi, ℓf, mf, ℓo, mo)
                end
            end
            CG_mats[(ℓo, mo)] = cu(CG_mat)
        end
    end
    
    CLayer(F_NN, CG_mats, ℓi, ℓf, ℓms)
end

# Forward pass
"""
V indexed by [mi, b]
"""

#=
function (C::CLayer)(rr, V)
    n_points, _, _, n_samples = size(rr)

    F_out = permutedims(C.F(rr), (1, 4, 2, 3))
    F_reshape = reshape(F_out, n_points, 2C.ℓf + 1, :)

    V_repeat = repeat(V, inner=(1, 1, n_points))
    F_tilde = batched_mul(V_repeat, F_reshape)

    # TODO Make this general
    # For now just assume one CG_mat
    CG_mat = CuArray(C.CG_mats[C.ℓms[1]])

    L_tilde = CG_mat .* F_tilde

    # Steps using a lot of memory atm
    L = sum(L_tilde, dims=(1, 2))#[1,1,:]
    reshape(L, n_points, n_samples)
end
=#


# Trying to use tensorcast
function (C::CLayer)(rr, V)
    n_points, _, _, n_samples = size(rr)

    F_out = C.F(rr)
    
    # Using Einstein summation convention for brevity
    # Speed seems comparable
    @reduce F_tilde[mi, mf, a, γ] := sum(b) V[mi, b, γ] * F_out[b, a, γ, mf]

    # TODO Make this general
    # For now just assume one CG_mat
    CG_mat = C.CG_mats[C.ℓms[1]]

    L_tilde = CG_mat .* F_tilde

    @reduce L[a, γ] := sum(mi, mf) L_tilde[mi, mf, a, γ]
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
