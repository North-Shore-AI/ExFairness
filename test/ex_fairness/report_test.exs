defmodule ExFairness.ReportTest do
  use ExUnit.Case, async: true
  doctest ExFairness.Report

  alias ExFairness.Report

  describe "generate/4" do
    setup do
      predictions =
        Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      {:ok, predictions: predictions, labels: labels, sensitive: sensitive}
    end

    test "generates report with all metrics", %{
      predictions: predictions,
      labels: labels,
      sensitive: sensitive
    } do
      report =
        Report.generate(predictions, labels, sensitive,
          metrics: [:demographic_parity, :equalized_odds, :equal_opportunity, :predictive_parity]
        )

      assert Map.has_key?(report, :demographic_parity)
      assert Map.has_key?(report, :equalized_odds)
      assert Map.has_key?(report, :equal_opportunity)
      assert Map.has_key?(report, :predictive_parity)
      assert Map.has_key?(report, :overall_assessment)
    end

    test "generates report with subset of metrics", %{
      predictions: predictions,
      labels: labels,
      sensitive: sensitive
    } do
      report =
        Report.generate(predictions, labels, sensitive,
          metrics: [:demographic_parity, :equalized_odds]
        )

      assert Map.has_key?(report, :demographic_parity)
      assert Map.has_key?(report, :equalized_odds)
      refute Map.has_key?(report, :equal_opportunity)
      refute Map.has_key?(report, :predictive_parity)
    end

    test "includes overall assessment", %{
      predictions: predictions,
      labels: labels,
      sensitive: sensitive
    } do
      report =
        Report.generate(predictions, labels, sensitive,
          metrics: [:demographic_parity, :equalized_odds]
        )

      assert Map.has_key?(report, :overall_assessment)
      assert is_binary(report.overall_assessment)
    end

    test "counts passed and failed metrics", %{
      predictions: predictions,
      labels: labels,
      sensitive: sensitive
    } do
      report =
        Report.generate(predictions, labels, sensitive,
          metrics: [:demographic_parity, :equalized_odds, :equal_opportunity, :predictive_parity]
        )

      assert Map.has_key?(report, :passed_count)
      assert Map.has_key?(report, :failed_count)
      assert Map.has_key?(report, :total_count)
      assert report.total_count == 4
    end

    test "defaults to all available metrics when none specified", %{
      predictions: predictions,
      labels: labels,
      sensitive: sensitive
    } do
      report = Report.generate(predictions, labels, sensitive)

      # Should include all implemented metrics
      assert Map.has_key?(report, :demographic_parity)
      assert Map.has_key?(report, :equalized_odds)
      assert Map.has_key?(report, :equal_opportunity)
      assert Map.has_key?(report, :predictive_parity)
    end

    test "passes through options to metrics", %{
      predictions: predictions,
      labels: labels,
      sensitive: sensitive
    } do
      report =
        Report.generate(predictions, labels, sensitive,
          metrics: [:demographic_parity],
          threshold: 0.05
        )

      assert report.demographic_parity.threshold == 0.05
    end
  end

  describe "to_markdown/1" do
    setup do
      predictions =
        Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      report = Report.generate(predictions, labels, sensitive, metrics: [:demographic_parity])
      {:ok, report: report}
    end

    test "returns markdown string", %{report: report} do
      markdown = Report.to_markdown(report)

      assert is_binary(markdown)
      assert String.contains?(markdown, "# Fairness Report")
      assert String.contains?(markdown, "## Overall Assessment")
    end

    test "includes metric results", %{report: report} do
      markdown = Report.to_markdown(report)

      assert String.contains?(markdown, "Demographic Parity")
    end

    test "includes table with results", %{report: report} do
      markdown = Report.to_markdown(report)

      assert String.contains?(markdown, "|")
      assert String.contains?(markdown, "Metric")
      assert String.contains?(markdown, "Passes")
    end
  end

  describe "to_json/1" do
    setup do
      predictions =
        Nx.tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])

      labels = Nx.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
      sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

      report = Report.generate(predictions, labels, sensitive, metrics: [:demographic_parity])
      {:ok, report: report}
    end

    test "returns valid JSON string", %{report: report} do
      json = Report.to_json(report)

      assert is_binary(json)
      assert {:ok, _decoded} = Jason.decode(json)
    end

    test "includes all report fields", %{report: report} do
      json = Report.to_json(report)
      {:ok, decoded} = Jason.decode(json)

      assert Map.has_key?(decoded, "overall_assessment")
      assert Map.has_key?(decoded, "demographic_parity")
    end
  end
end
