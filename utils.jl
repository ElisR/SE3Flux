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
Replace every Float64 in an expression with Float32, and return its equivalent function.
Bodge by using regex to replace floats e.g. `4.263 -> 4.263f0`.
This should stop returned functions from promoting Float32 arguments to Float64.
"""
function convert_expr_to_F32(Y_expr::Expr)::Function
    # Tidying string by removing filler.
    str = Base.remove_linenums!(Y_expr) |> repr
    str_F32 = replace(str, r"\d+\.\d+" => s"\g<0>f0")
    str_F32 |> Meta.parse |> eval |> eval
end

function generate_Yℓms(ℓ::Int)
    @variables θ::Real, ϕ::Real
    Ys_sym = computeYlm(θ, ϕ; lmax=ℓ, SHType=SphericalHarmonics.RealHarmonics())
    keys = [(ℓ, m) for m in -ℓ:ℓ]
    # Ys_sym.modes[1] |> collect # to access available keys

    [convert_expr_to_F32(build_function((-1)^m * Ys_sym[(ℓ, m)] |> simplify, θ, ϕ)) for (ℓ, m) in keys]
end

"""
Defining function for Clebsch-Gordan matrices.
Employing a change of basis to work for real spherical harmonics.
"""
@memoize function generate_CG_matrices(ℓi::Int, ℓf::Int, ℓo::Int)
    # Define basis rotation
    Ai, Af, Ao = basis_rotation.((ℓi, ℓf, ℓo))

    CG_mat = zeros(Float64, (2ℓi + 1, 2ℓf + 1, 2ℓo + 1))
    for (i_o, mo) in enumerate(-ℓo:ℓo)
        for (i_f, mf) in enumerate(-ℓf:ℓf)
            for (i_i, mi) in enumerate(-ℓi:ℓi)
                CG_mat[i_i, i_f, i_o] = cg(ℓi, mi, ℓf, mf, ℓo, mo)
            end
        end
    end

    @reduce CG_real[Mi, Mf, Mo] := sum(mi, mf, mo) CG_mat[mi, mf, mo] * Ao[Mo, mo] * Ai'[mi, Mi] * Af'[mf, Mf]
    #@assert CG_real |> imag .|> Float32 |> maximum ≈ 0
    CG_real |> real .|> Float32
end

"""
Function for generating the basis rotation from complex to real spherical harmonics.
Outputs a matrix, which is the matrix to invert
"""
@memoize function basis_rotation(ℓ::Int)
    A = zeros(ComplexF64, (2ℓ + 1, 2ℓ + 1))
    ind(m) = m + ℓ + 1
    for m in -ℓ:ℓ
        CS = (-1)^m # Condon-Shortley Phase
        if m < 0
            A[ind(m), ind(m)] = 1im / √2
            A[ind(m), ind(-m)] = -CS * 1im / √2
        elseif m > 0
            A[ind(m), ind(m)] = CS * 1 / √2
            A[ind(m), ind(-m)] = 1 / √2
        else
            A[ind(m), ind(m)] = 1
        end
    end

    # Adding phase that supposedly makes CG real eventually
    (-1im)^ℓ * A
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
"""
# In the main programme, tuple_connect should only receive multiple tuples, each of which has many vectors of CuArrays

function tuple_connect(xs...)
    Tuple(vcat(x...) for x in zip(xs...))
end