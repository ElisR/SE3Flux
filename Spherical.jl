# Code for calculating Spherical Harmonics
# Copied from FastTransforms.jl for now before I reimplment
using Memoize

"""
Compute a typed 0.5.
"""
half(x::Number) = oftype(x,0.5)
half(x::Integer) = half(float(x))
half(::Type{T}) where {T<:Number} = convert(T,0.5)
half(::Type{T}) where {T<:Integer} = half(AbstractFloat)

"""
Compute a typed 2.
"""
two(x::Number) = oftype(x,2)
two(::Type{T}) where {T<:Number} = convert(T,2)

"""
Pointwise evaluation of real orthonormal spherical harmonic:
```math
Y_\\ell^m(\\theta,\\varphi) = (-1)^{|m|}\\sqrt{(\\ell+\\frac{1}{2})\\frac{(\\ell-|m|)!}{(\\ell+|m|)!}} P_\\ell^{|m|}(\\cos\\theta) \\sqrt{\\frac{2-\\delta_{m,0}}{2\\pi}} \\left\\{\\begin{array}{ccc} \\cos m\\varphi & {\\rm for} & m \\ge 0,\\\\ \\sin(-m\\varphi) & {\\rm for} & m < 0.\\end{array}\\right.
```
"""
sphevaluate(θ, L, M) = sphevaluatepi(θ/π, L, M)

function sphevaluatepi(θ::Number, L::Integer, M::Integer)
    # This seems to differ from SE(3) Transformers implementation

    ret = one(θ)/sqrt(two(θ))
    if M < 0 M = -M end
    c, s = cospi(θ), sinpi(θ)
    for m = 1:M
        ret *= sqrt((m+half(θ))/m)*s
    end
    tc = two(c)*c

    if L == M
        return ret
    elseif L == M+1
        return sqrt(two(θ)*M+3)*c*ret
    else
        temp = ret
        ret *= sqrt(two(θ)*M+3)*c
        for l = M+1:L-1
            ret, temp = (sqrt(l+half(θ))*tc*ret - sqrt((l-M)*(l+M)/(l-half(θ)))*temp)/sqrt((l-M+1)*(l+M+1)/(l+3half(θ))), ret
        end
        return ret
    end
end

# Clebsch-Gordan Naive Calculation
# Taken from https://github.com/cortner/SphericalHarmonics.jl/blob/master/src/clebschgordan.jl
"""
`cg1(j1, m1, j2, m2, J, M, T=Float64)` : A reference implementation of
Clebsch-Gordon coefficients based on
https://hal.inria.fr/hal-01851097/document
Equation (4-6)
This heavily uses BigInt and BigFloat and should therefore not be employed
for performance critical tasks.
"""
@memoize function cg1(j1, m1, j2, m2, J, M)
   if (M != m1 + m2) || !(abs(j1-j2) <= J <= j1 + j2)
      return zero(T)
   end

   N = (2*J+1) *
       factorial(big(j1+m1)) * factorial(big(j1-m1)) *
       factorial(big(j2+m2)) * factorial(big(j2-m2)) *
       factorial(big(J+M)) * factorial(big(J-M)) /
       factorial(big( j1+j2-J)) /
       factorial(big( j1-j2+J)) /
       factorial(big(-j1+j2+J)) /
       factorial(big(j1+j2+J+1))

   G = big(0)
   # 0 ≦ k ≦ j1+j2-J
   # 0 ≤ j1-m1-k ≤ j1-j2+J   <=>   j2-J-m1 ≤ k ≤ j1-m1
   # 0 ≤ j2+m2-k ≤ -j1+j2+J  <=>   j1-J+m2 ≤ k ≤ j2+m2
   lb = (0, j2-J-m1, j1-J+m2)
   ub = (j1+j2-J, j1-m1, j2+m2)
   for k in maximum(lb):minimum(ub)
      G += (-1)^k *
           binomial(big( j1+j2-J), big(k)) *
           binomial(big( j1-j2+J), big(j1-m1-k)) *
           binomial(big(-j1+j2+J), big(j2+m2-k))
   end

   return Float64(sqrt(N) * G)
end