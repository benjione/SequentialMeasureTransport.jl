using Documenter, SequentialMeasureTransport
using SequentialMeasureTransport.Statistics

makedocs(sitename="SequentialMeasureTransport.jl",
        authors="Benjamin Zanger",
        pages=[
            "Home" => "index.md",
            "Mathematical Background" => "man/math_background.md",
            "API" => [
                "SequentialMeasureTransport" => "api/SMT.md",
                "Statistics" => "api/statistics.md",
            ]
        ],
        # modules = [SequentialMeasureTransport],
        )