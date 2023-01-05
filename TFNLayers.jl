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
    Ys::Vector{Function} # SH for this ℓf
    ℓf::Int # Filter angular momentum
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
    rr_radial = rr[:,1]'
    R_out = F.R(rr_radial)

    # Multiply by SH components
    Y_out = reduce(hcat, [Y.(rr[:,2], rr[:,3]) for Y in F.Ys])
    R_out' .* Y_out
end

Flux.@functor FLayer (R,)

#=
Flux.@functor E3ConvLayer
struct E3ConvLayer
    # Learned radial functions
    # TODO Check if putting it in a dictionary is bad
    Rs::Dict{Tuple{Unsigned, Int}, Chain}

    # Store precomputed SH functions
    # Hopefully allowed by Flux.jl, and specifiable as constants
    # Use (ℓ, m) as dictionary keys
    Ys::Dict{Tuple{Unsigned, Int}, Function}

    # Store CGs[(j1, m1, j2, m2, J, M)] = <j1 m1 j2 m2 | j1 j2; J M>
    CGs::Dict{Tuple{Unsigned, Int, Unsigned, Int, Unsigned, Int}, Float32} # Clebsch-Gordan coefficients
end

"""
    E3ConvLayer(S::Unsigned, n_ℓs::Vector{Unsigned}, filter_ℓ::Unsigned)

Constructor for `S` points, `[n_ℓ]` channels for each `ℓ`, `filter_ℓ` is angular momentum of filter.

`n_ℓs` is a vector `[n_0, n_1, ..., n_L]` where `ℓ=L` is the largest nonzero angular momentum feature.
(Preceding `n_i=0` entries must be present.)
Spends time generating dictionary of SH functions.
"""
function E3ConvLayer(S::Unsigned, n_ℓs::Vector{Unsigned}, ℓ_filter::Unsigned)
    # TODO There should be a list of filters

    # Calculate the maximum SH needed
    max_ℓ = length(n_ℓs) + ℓ_filter

    # Number of learned R functions needed
    n_channels = sum(n_ℓs) # This is wrong

    # Generate R chain with correct dimensions
    # Default to two hidden layers

    # Generate SHs using Symbolics package
    # Slow because symbolic algebra, but only done once
    @variables θ, ϕ
    Ys_sym = computeYlm(θ, ϕ; lmax=2, SHType=SphericalHarmonics.RealHarmonics())

    # Create dictionary of fast Y[ℓ, m](θ, ϕ) functions
    Ys = Dict(key => (eval ∘ build_function)(Ys_sym[key] |> simplify, θ, ϕ)
            for key in Ys_sym.modes[1] |> collect)

    # Calculate all necessary CG coefficients
    # TODO Make sure all correct entries are filled out
    # TODO Use all symmetries
    # TODO Maybe put this in a separate function
    CGs = Dict()
    ℓ1 = ℓ_filter
    for ℓ in 0:max_ℓ
        for m in -ℓ:ℓ
            for m1 in -ℓ1:ℓ1
                for ℓ2 in [ℓ_nz-1 for (ℓ_nz, _) in filter(x -> x[2] > 0, enumerate(n_ℓs) |> collect)]
                    for m2 in -ℓ2:ℓ2
                        spec = (ℓ1, m1, ℓ2, m2, ℓ, m)
                        CGs[spec] = cg(spec...)

    # Use default constructor
    E3ConvLayer(Rs, Ys, CGs)
end



"""
    (l::E3ConvLayer)(r, V)

Forward pass with distances and features.
"""
function (l::E3ConvLayer)(r, V)
    # Create all distance pairings

    # Convert to spherical harmonics
    # Should be done for all pairs of distances

    # For all channels, pass every distance through R and sum over points
    # Superpose all according to Clebsch-Gordan coefficients
end
=#