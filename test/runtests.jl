using PSDModels
using LinearAlgebra
using DomainSets
using Test

@testset "internal utility test" begin
    include("utils_test.jl")
end

@testset "PSDModelKernel test" begin
    include("PSDKernel_test.jl")
end

@testset "PSDModelFM test" begin
    include("PSDFM_test.jl")
end

@testset "PSDModel polynomial test" begin
    include("PSDPolynomial_test.jl")
end

@testset "PSDModel tensor polynomial test" begin
    include("PSDTensorPolynomial_test.jl")
end