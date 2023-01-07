using Memoize

# Clebsch-Gordan Naive Calculation
# Taken from https://github.com/cortner/SphericalHarmonics.jl/blob/master/src/clebschgordan.jl
"""
`cg(j1, m1, j2, m2, J, M)` : A reference implementation of
Clebsch-Gordon coefficients based on
https://hal.inria.fr/hal-01851097/document
Equation (4-6)
This heavily uses BigInt and BigFloat and should therefore not be employed
for performance critical tasks.
"""
@memoize function cg(j1, m1, j2, m2, J, M)
   if (M != m1 + m2) || !(abs(j1-j2) <= J <= j1 + j2)
      return zero(Float32)
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

   return Float32(sqrt(N) * G)
end