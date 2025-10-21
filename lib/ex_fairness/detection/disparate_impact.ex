defmodule ExFairness.Detection.DisparateImpact do
  @moduledoc """
  Disparate Impact detection using the 80% rule (4/5ths rule).

  The 80% rule is a legal guideline from the EEOC (Equal Employment Opportunity
  Commission) used to determine if there is adverse impact in employment decisions.

  ## The 80% Rule

  A selection rate for any group that is less than 80% (4/5ths) of the rate for
  the group with the highest selection rate is generally regarded as evidence of
  adverse impact.

      Ratio = (Selection Rate for Group with Lower Rate) / (Selection Rate for Group with Higher Rate)

  If Ratio ≥ 0.8, the process passes the 80% rule.
  If Ratio < 0.8, there may be disparate impact.

  ## Legal Context

  This is a legal standard, not just a statistical measure. It's used in:
  - Employment discrimination cases (hiring, promotion, termination)
  - Lending decisions
  - Educational admissions
  - Housing decisions

  ## Limitations

  - The 80% rule is a guideline, not an absolute legal requirement
  - Statistical significance should also be considered
  - Small sample sizes may produce unreliable ratios
  - Does not prove discrimination, only suggests further investigation

  ## References

  - EEOC Uniform Guidelines on Employee Selection Procedures (1978)
  - https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Detection.DisparateImpact.detect(predictions, sensitive)
      iex> result.passes_80_percent_rule
      true

  """

  alias ExFairness.Utils
  alias ExFairness.Validation

  @default_min_per_group 10
  @eighty_percent_threshold 0.8

  @type result :: %{
          group_a_rate: float(),
          group_b_rate: float(),
          ratio: float(),
          passes_80_percent_rule: boolean(),
          interpretation: String.t()
        }

  @doc """
  Detects disparate impact using the 80% rule.

  ## Parameters

    * `predictions` - Binary predictions tensor (0 or 1)
    * `sensitive_attr` - Binary sensitive attribute tensor (0 or 1)
    * `opts` - Options:
      * `:min_per_group` - Minimum samples per group for validation (default: 10)

  ## Returns

  A map containing:
    * `:group_a_rate` - Selection rate for group A
    * `:group_b_rate` - Selection rate for group B
    * `:ratio` - Disparate impact ratio (min rate / max rate)
    * `:passes_80_percent_rule` - Whether ratio ≥ 0.8
    * `:interpretation` - Plain language explanation

  ## Examples

      iex> predictions = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
      iex> sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      iex> result = ExFairness.Detection.DisparateImpact.detect(predictions, sensitive)
      iex> result.passes_80_percent_rule
      false

  """
  @spec detect(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: result()
  def detect(predictions, sensitive_attr, opts \\ []) do
    min_per_group = Keyword.get(opts, :min_per_group, @default_min_per_group)

    # Validate inputs
    Validation.validate_predictions!(predictions)

    Validation.validate_matching_shapes!([predictions, sensitive_attr], [
      "predictions",
      "sensitive_attr"
    ])

    Validation.validate_sensitive_attr!(sensitive_attr, min_per_group: min_per_group)

    # Compute selection rates for both groups
    {rate_a, rate_b} = Utils.group_positive_rates(predictions, sensitive_attr)
    rate_a_num = Nx.to_number(rate_a)
    rate_b_num = Nx.to_number(rate_b)

    # Compute disparate impact ratio (min/max)
    ratio = compute_disparate_impact_ratio(rate_a_num, rate_b_num)

    # Determine if passes 80% rule
    passes = ratio >= @eighty_percent_threshold

    # Generate interpretation
    interpretation = generate_interpretation(rate_a_num, rate_b_num, ratio, passes)

    %{
      group_a_rate: rate_a_num,
      group_b_rate: rate_b_num,
      ratio: ratio,
      passes_80_percent_rule: passes,
      interpretation: interpretation
    }
  end

  # Compute disparate impact ratio as min/max to detect disparity in either direction
  @spec compute_disparate_impact_ratio(float(), float()) :: float()
  defp compute_disparate_impact_ratio(rate_a, rate_b) do
    cond do
      # Both zero - no disparity
      rate_a == 0.0 and rate_b == 0.0 -> 1.0
      # Both one - no disparity
      rate_a == 1.0 and rate_b == 1.0 -> 1.0
      # One is zero - maximum disparity
      rate_a == 0.0 or rate_b == 0.0 -> 0.0
      # Normal case - compute min/max
      true -> min(rate_a, rate_b) / max(rate_a, rate_b)
    end
  end

  @spec generate_interpretation(float(), float(), float(), boolean()) :: String.t()
  defp generate_interpretation(rate_a, rate_b, ratio, passes) do
    rate_a_pct = Float.round(rate_a * 100, 1)
    rate_b_pct = Float.round(rate_b * 100, 1)
    ratio_pct = Float.round(ratio * 100, 1)

    base_msg =
      "Group A selection rate: #{rate_a_pct}%. " <>
        "Group B selection rate: #{rate_b_pct}%. " <>
        "Disparate impact ratio: #{ratio_pct}% (#{Float.round(ratio, 3)})."

    legal_msg =
      if passes do
        " This PASSES the 80% rule (4/5ths rule). " <>
          "No evidence of disparate impact under EEOC guidelines."
      else
        " This FAILS the 80% rule (4/5ths rule). " <>
          "This may indicate disparate impact under EEOC guidelines and warrants further investigation. " <>
          "Note: The 80% rule is a guideline, not an absolute legal standard."
      end

    base_msg <> legal_msg
  end
end
