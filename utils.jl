using Symbolics
using SphericalHarmonics
using Memoize

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
            out[i,j,:,:] .= @view(rs[j,:,:]) .- @view(rs[i,:,:])
        end
    end

    out
end

"""
Utility `connection` function for `Flux.Parallel` to output tuple from input tuple.
"""
function triv_connect(x...)
    x
end

"""
Utility `connection` function for `Flux.Parallel` to join two tuples of vectors into one tuple of concatenated vectors from each.
Assumes both have the same tuple length, with one possibly padded with empty vectors `[]` (of the correct type).
Should naturally perform the concatenation mentioned in TFN paper.
In the main programme, tuple_connect should only receive multiple tuples, each of which has many vectors of CuArrays.
"""
function tuple_connect(xs...)
    Tuple(vcat(x...) for x in zip(xs...))
end