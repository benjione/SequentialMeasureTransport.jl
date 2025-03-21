using SequentialMeasureTransport
import SequentialMeasureTransport as SMT
using LinearAlgebra
using ApproxFun
using Distributions
using Test
import SCS

@testset "internal utility test" begin
    include("utils_test.jl")
end

@testset "Tensorizers test" begin
    include("Tensorizer_test.jl")
end

@testset "TensorPolynomial tests" begin
    include("TensorPolynomial_test.jl")
end

# @testset "PSDModelKernel test" begin
#     include("PSDKernel_test.jl")
# end

@testset "PSDModelFM test" begin
    include("PSDFM_test.jl")
end

@testset "PSDModelPolynomial test" begin
    include("PSDPolynomial_test.jl")
end

@testset "Statistics test" begin
    include("Statistics_test.jl")
end

@testset "Sampler test" begin
    include("Sampler/Sampler_test.jl")
end