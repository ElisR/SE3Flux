using BenchmarkTools
using Test
using SphericalHarmonics
using Memoize
using Symbolics
using TensorCast

include("../Spherical.jl")
include("../utils.jl")

@testset "Wikipedia Clebsch-Gordan Coefficients" begin
    # Some examples from Wikipedia
    # TODO Get SymPy.jl comparison to work

    # cg(j1, m1, j2, m2, J, M) = <j1 m1 j2 m2; J M>
    @test cg(1, 1, 1, -1, 2, 0; T = Float64) == √(1 / 6)
    @test cg(1, 0, 1, 0, 2, 0;T = Float64) == √(2 / 3)
    @test cg(1, 0, 1, 0, 0, 0;T = Float64) == -√(1 / 3)
    @test cg(1, 0, 1, 0, 1, 0;T = Float64) == 0

    @test cg(2, 1, 1, 0, 3, 1;T = Float64) == √(8 / 15)
end;

@testset "Spherical Harmonics" begin
    Y00(θ, ϕ) = 0.5 * sqrt(1/π)
    Y1m1(θ, ϕ) = 0.5 * sqrt(3/(2*π))*sin(θ)*exp(-im*ϕ)
    Y10(θ, ϕ) = 0.5 * sqrt(3/π)*cos(θ)
    Y11(θ, ϕ) = -0.5 * sqrt(3/(2*π))*sin(θ)*exp(im*ϕ)

    # Real version
    function Y1mR(θ, ϕ, m::Integer)
        ans = Y10(θ, ϕ) |> real
        if m > 0
            ans = -(Y1m1(θ, ϕ) + (-1)^m * Y11(θ, ϕ)) / √2 |> real
        elseif m < 0
            ans = -im * (Y1m1(θ, ϕ) - (-1)^m * Y11(θ, ϕ)) / √2 |> real
        end

        ans
    end

    n_samples = 10
    for n = 1:n_samples
        θ = rand() * π
        ϕ = (rand()-0.5) * 2*π
        YsR = computeYlm(θ, ϕ; lmax=1, SHType=SphericalHarmonics.RealHarmonics())

        @test YsR[(0, 0)] ≈ Y00(θ, ϕ)
        @test YsR[(1, -1)] ≈ Y1mR(θ, ϕ, -1)
        @test YsR[(1, 1)] ≈ Y1mR(θ, ϕ, 1)
        @test YsR[(1, 0)] ≈ Y1mR(θ, ϕ, 0)
        #@test Ys[(1, -1)] ≈ Y1m1(θ, ϕ)
    end
end;