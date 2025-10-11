defmodule ExFairness.MixProject do
  use Mix.Project

  @version "0.1.0"
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
      name: "ExFairness"
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.7"},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false}
    ]
  end

  defp description do
    "Fairness and bias detection library for Elixir AI/ML systems. Provides comprehensive fairness metrics, bias detection algorithms, and mitigation techniques."
  end

  defp package do
    [
      name: "ex_fairness",
      description: description(),
      files: ~w(lib mix.exs README.md CHANGELOG.md LICENSE),
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Online documentation" => "https://hexdocs.pm/ex_fairness"
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
      extras: ["README.md", "CHANGELOG.md"],
      assets: %{"assets" => "assets"},
      logo: "assets/ex_fairness.svg",
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
