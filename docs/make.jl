using Documenter, SequentialMeasureTransport
using SequentialMeasureTransport.Statistics

using Literate


const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

examples = [
  "1D_density_estimation_single_map.jl",
]


for example in examples
  withenv("GITHUB_REPOSITORY" => "FourierFlows/GeophysicalFlowsDocumentation") do
    example_filepath = joinpath(EXAMPLES_DIR, example)
    withenv("JULIA_DEBUG" => "Literate") do
      Literate.markdown(example_filepath, OUTPUT_DIR;
                        flavor = Literate.DocumenterFlavor(), execute = true)
    end
  end
end


makedocs(sitename="SequentialMeasureTransport.jl",
        authors="Benjamin Zanger",
        pages=[
            "Home" => "index.md",
            "Mathematical Background" => "man/math_background.md",
            "Examples" => [
                "1D Density Estimation" => "literated/1D_density_estimation_single_map.md",
            ],
            "API" => [
                "SequentialMeasureTransport" => "api/SMT.md",
                "Statistics" => "api/statistics.md",
            ]
        ],
        # modules = [SequentialMeasureTransport],
        )