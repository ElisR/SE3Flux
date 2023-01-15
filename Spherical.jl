using Memoize

"""
`cg(j1, m1, j2, m2, J, M)` : A reference implementation of
Clebsch-Gordan coefficients based on
https://hal.inria.fr/hal-01851097/document and Equations (4-6)
This uses BigInt and BigFloat and should not be employed
for performance critical tasks, which is fine because they are only calculated when initialising the network.
Function taken from https://github.com/cortner/SphericalHarmonics.jl/blob/master/src/clebschgordan.jl
"""
@memoize function cg(j1, m1, j2, m2, J, M; T::Type = Float32)
   if (M != m1 + m2) || !(abs(j1 - j2) <= J <= j1 + j2)
      return zero(T)
   end

   N = (2*J+1) *
       factorial(big(j1 + m1)) * factorial(big(j1 - m1)) *
       factorial(big(j2 + m2)) * factorial(big(j2 - m2)) *
       factorial(big(J + M)) * factorial(big(J - M)) /
       factorial(big(j1 + j2 - J)) /
       factorial(big(j1 - j2 + J)) /
       factorial(big(-j1 + j2 + J)) /
       factorial(big(j1 + j2 + J + 1))

   G = big(0)
   # 0 ≦ k ≦ j1+j2-J
   # 0 ≤ j1-m1-k ≤ j1-j2+J   <=>   j2-J-m1 ≤ k ≤ j1-m1
   # 0 ≤ j2+m2-k ≤ -j1+j2+J  <=>   j1-J+m2 ≤ k ≤ j2+m2
   lb = (0, j2 - J - m1, j1 - J + m2)
   ub = (j1 + j2 - J, j1 - m1, j2 + m2)
   for k in maximum(lb):minimum(ub)
      G += (-1)^k *
           binomial(big(j1 + j2 -J), big(k)) *
           binomial(big(j1 - j2 +J), big(j1 - m1 - k)) *
           binomial(big(-j1 + j2 +J), big(j2 + m2 - k))
   end

   return convert(T, √N * G)
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

@memoize function generate_Yℓms(ℓ::Int)
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
        CS = (-1)^m # Condon-Shortley phase
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
    # Adding phase that makes CG real
    (-1im)^ℓ * A
end