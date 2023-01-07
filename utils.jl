using Symbolics
using SphericalHarmonics

"""
    `acos_nonan(a::T)`

Function that replaces `NaN` (encountered if `r = 0` in spherical coordinate conversion) with `0`.
`0` has same type as input `Nan`.
Technically not type-safe since if `a` is an Integer, `acos(a)` will not return an Integer.
"""
function acos_nonan(a::T) where T
    isnan(a) ? zero(T) : acos(a)
end

"""
    `cart_to_sph(xss::Array{T, 4})`

Convert pairwise cartesian vectors into spherical coordinates.
3rd dimension is `[r, θ, ϕ]`
"""
function cart_to_sph(xss::Array{T, 4}) where T
    xs = @view xss[:,:,1,:]
    ys = @view xss[:,:,2,:]
    zs = @view xss[:,:,3,:]

    out = similar(xss)

    @. out[:,:,1,:] = √(xs^2 + ys^2 + zs^2) 
    @. out[:,:,2,:] = acos_nonan(zs / out[:,:,1,:])
    @. out[:,:,3,:] = @. atan(ys, xs)

    out
end

"""
    `pairwise_rs(rs::Array{T, 3})`

Calculate the mutual distances between all pairs.
Data arranged as `[a, i, γ]`, where `γ` is sample, `a` is point index in point cloud, `i ∈ (x,y,z)` are cartesian indices.
"""
function pairwise_rs(rs::Array{T, 3}) where T
    shape = size(rs)
    out = zeros(T, (shape[1], shape[1], shape[2], shape[3]))

    for i in 1:shape[1]
        for j in 1:shape[1]
            out[i,j,:,:] .= @view(rs[i,:,:]) .- @view(rs[j,:,:])
        end
    end

    out
end

"""
Replace every Float64 in an expression with Float32, and return its equivalent function.
Bodge by using regex to replace floats e.g. `4.263 -> 4.263f0`.
This should stop returned functions from promoting Float32 arguments to Float64.
"""
function convert_expr_to_F32(Y_expr::Expr)
    # Tidying string by removing filler.
    str = Base.remove_linenums!(Y_expr) |> repr
    str_F32 = replace(str, r"\d+\.\d+" => s"\g<0>f0")
    Y = str_F32 |> Meta.parse |> eval |> eval

    # Adding some type annotation
    function Y_F32(θ, ϕ)::Float32
        Y(θ, ϕ)
    end

    return Y_F32
end

function generate_Yℓms(ℓ::Int)
    @variables θ::Real, ϕ::Real
    Ys_sym = computeYlm(θ, ϕ; lmax=ℓ, SHType=SphericalHarmonics.RealHarmonics())
    keys = [(ℓ, m) for m in -ℓ:ℓ]

    [convert_expr_to_F32(build_function(Ys_sym[key] |> simplify, θ, ϕ)) for key in keys]
end

#=
# Old function for conversion 
"""
Convert cartesian vectors into spherical coordinates.
Columns are [r, θ, ϕ]
"""
function cart_to_sph_old(xs)
    rs = @. √(xs[:,1]^2 + xs[:,2]^2 + xs[:,3]^2) 
    θs = @. acos(xs[:,3] / rs)
    ϕs = @. atan(xs[:,2], xs[:,1])

    [rs θs ϕs]
end
=#