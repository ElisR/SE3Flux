using Flux

# SH and CG coefficients
using Symbolics
using SphericalHarmonics
include("Spherical.jl")

# -------------------
# F layer which sums
# -------------------

struct FLayer
    R::Chain # Radial NN
    Ys::Vector{Function} # SH functions for this ℓf
    ℓf::Int # Filter angular momentum # TODO Consider removing
end

function FLayer(Ys::Vector{Function})
    # Will later allow for custom spec
    R = Chain(
        Dense(1 => 5, relu),
        Dense(5 => 5, relu),
        Dense(5 => 1, relu)
    )

    ℓf = (length(Ys) - 1) ÷ 2
    FLayer(R, Ys, ℓf)
end

function (F::FLayer)(rr)
    # Apply R to the input radii
    rr_radial_mat = @view rr[:, :, 1, :]
    shape = size(rr_radial_mat)
    rr_radials_vec = reshape(rr_radial_mat, 1, :)

    R_out_vec = F.R(rr_radials_vec)
    R_out = reshape(R_out_vec, shape)

    # Multiply by SH components
    θs = @view rr[:,:,2,:]
    ϕs = @view rr[:,:,3,:]
    Y_out = Flux.batch([Y.(θs, ϕs) for Y in F.Ys])
    
    R_out .* Y_out
end

Flux.@functor FLayer (R,)

#=
"""
    E3ConvLayer(S::Unsigned, n_ℓs::Vector{Unsigned}, filter_ℓ::Unsigned)

Constructor for `S` points, `[n_ℓ]` channels for each `ℓ`, `filter_ℓ` is angular momentum of filter.

`n_ℓs` is a vector `[n_0, n_1, ..., n_L]` where `ℓ=L` is the largest nonzero angular momentum feature.
(Preceding `n_i=0` entries must be present.)
Spends time generating dictionary of SH functions.
"""
function E3ConvLayer(S::Unsigned, n_ℓs::Vector{Unsigned}, ℓ_filter::Unsigned)
