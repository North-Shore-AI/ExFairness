defmodule ExFairness.MixProject do
  use Mix.Project

  @version "0.5.0"
  @source_url "https://github.com/North-Shore-AI/ExFairness"

  def project do
    [
      app: :ex_fairness,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package(),
      docs: docs(),
      source_url: @source_url,
      homepage_url: @source_url,
      name: "ExFairness",
      # Quality gates
      elixirc_options: [warnings_as_errors: true],
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.detail": :test,
        "coveralls.html": :test,
        "coveralls.json": :test
      ],
      dialyzer: [
        plt_add_apps: [:mix, :ex_unit],
        plt_core_path: "priv/plts",
        plt_file: {:no_warn, "priv/plts/dialyzer.plt"}
      ]
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      # Core dependencies
      {:crucible_framework, "~> 0.5.0"},
      {:crucible_ir, "~> 0.2.1"},
      {:jason, "~> 1.4"},
      {:nx, "~> 0.7"},
      # Required by crucible_framework
      {:ecto_sql, "~> 3.11"},
      {:postgrex, ">= 0.21.1"},

      # Development and testing
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:excoveralls, "~> 0.18", only: :test},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:stream_data, "~> 1.0", only: :test}
    ]
  end

  defp description do
    "Fairness and bias detection library for Elixir AI/ML systems. Provides comprehensive fairness metrics, bias detection algorithms, and mitigation techniques."
  end

  defp package do
    [
      name: "ex_fairness",
      description: description(),
      files: ~w(lib mix.exs README.md CHANGELOG.md CONTRIBUTING.md LICENSE docs),
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Online documentation" => "https://hexdocs.pm/ex_fairness",
        "Technical Documentation" => "#{@source_url}/tree/main/docs/20251020"
      },
      maintainers: ["nshkrdotcom"]
    ]
  end

  defp docs do
    [
      main: "readme",
      name: "ExFairness",
      source_ref: "v#{@version}",
      source_url: @source_url,
      homepage_url: @source_url,
      extras: [
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "docs/architecture.md",
        "docs/metrics.md",
        "docs/algorithms.md",
        "docs/roadmap.md",
        "docs/20251020/future_directions.md",
        "docs/20251020/implementation_report.md",
        "docs/20251020/testing_and_qa_strategy.md"
      ],
      groups_for_extras: [
        "Getting Started": ["README.md", "CONTRIBUTING.md"],
        Architecture: [
          "docs/architecture.md",
          "docs/metrics.md",
          "docs/algorithms.md"
        ],
        Planning: [
          "docs/roadmap.md",
          "docs/20251020/future_directions.md"
        ],
        "Technical Reports": [
          "docs/20251020/implementation_report.md",
          "docs/20251020/testing_and_qa_strategy.md"
        ],
        Changelog: ["CHANGELOG.md"]
      ],
      groups_for_modules: [
        "Fairness Metrics": [
          ExFairness.Metrics.DemographicParity,
          ExFairness.Metrics.EqualizedOdds,
          ExFairness.Metrics.EqualOpportunity,
          ExFairness.Metrics.PredictiveParity,
          ExFairness.Metrics.Calibration
        ],
        Detection: [
          ExFairness.Detection.DisparateImpact
        ],
        Mitigation: [
          ExFairness.Mitigation.Reweighting
        ],
        Reporting: [
          ExFairness.Report
        ],
        Pipeline: [
          ExFairness.Stage,
          ExFairness.CrucibleStage
        ],
        Utilities: [
          ExFairness.Utils,
          ExFairness.Utils.Metrics,
          ExFairness.Utils.Bootstrap,
          ExFairness.Utils.StatisticalTests,
          ExFairness.Validation,
          ExFairness.Error
        ]
      ],
      assets: %{"assets" => "assets"},
      logo: "assets/ExFairness.svg",
      before_closing_head_tag: &mermaid_config/1
    ]
  end

  defp mermaid_config(:html) do
    """
    <script defer src="https://cdn.jsdelivr.net/npm/mermaid@10.2.3/dist/mermaid.min.js"></script>
    <script>
      let initialized = false;

      window.addEventListener("exdoc:loaded", () => {
        if (!initialized) {
          mermaid.initialize({
            startOnLoad: false,
            theme: document.body.className.includes("dark") ? "dark" : "default"
          });
          initialized = true;
        }

        let id = 0;
        for (const codeEl of document.querySelectorAll("pre code.mermaid")) {
          const preEl = codeEl.parentElement;
          const graphDefinition = codeEl.textContent;
          const graphEl = document.createElement("div");
          const graphId = "mermaid-graph-" + id++;
          mermaid.render(graphId, graphDefinition).then(({svg, bindFunctions}) => {
            graphEl.innerHTML = svg;
            bindFunctions?.(graphEl);
            preEl.insertAdjacentElement("afterend", graphEl);
            preEl.remove();
          });
        }
      });
    </script>
    """
  end

  defp mermaid_config(_), do: ""
end
