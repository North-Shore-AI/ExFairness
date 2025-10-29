#!/usr/bin/env elixir

# End-to-End Fairness Workflow Example
# This example demonstrates a complete fairness workflow from
# detection to mitigation to validation.

IO.puts("\n=== End-to-End Fairness Workflow ===\n")

IO.puts("Scenario: Building a fair loan approval system")
IO.puts("Following best practices for responsible AI")
IO.puts("")

:rand.seed(:exsplus, {42, 42, 42})

# Helper function to generate loan data
generate_loan_data = fn group_size, approval_rate, success_rate ->
  predictions = for _ <- 1..group_size, do: if(:rand.uniform() < approval_rate, do: 1, else: 0)

  labels =
    Enum.map(predictions, fn pred ->
      case pred do
        1 -> if :rand.uniform() < success_rate, do: 1, else: 0
        0 -> if :rand.uniform() < 0.25, do: 1, else: 0
      end
    end)

  {predictions, labels}
end

# ============================================================================
# STEP 1: INITIAL DATA ANALYSIS
# ============================================================================

IO.puts("=" |> String.duplicate(60))
IO.puts("STEP 1: INITIAL DATA ANALYSIS")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

# Generate initial training data with inherent bias
# Group A: 60% approval rate, 75% success rate for approved
# Group B: 40% approval rate, 75% success rate for approved
{group_a_preds, group_a_labels} = generate_loan_data.(200, 0.60, 0.75)
{group_b_preds, group_b_labels} = generate_loan_data.(200, 0.40, 0.75)

initial_predictions = Nx.tensor(group_a_preds ++ group_b_preds)
initial_labels = Nx.tensor(group_a_labels ++ group_b_labels)

initial_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {200}),
    Nx.broadcast(1, {200})
  ])

IO.puts("Dataset: 400 loan applications (200 per group)")
IO.puts("Training a baseline model...")

# ============================================================================
# STEP 2: BIAS DETECTION
# ============================================================================

IO.puts("\n")
IO.puts("=" |> String.duplicate(60))
IO.puts("STEP 2: COMPREHENSIVE BIAS DETECTION")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

# 2.1: Run comprehensive fairness report
IO.puts("2.1: Fairness Metrics Analysis")
IO.puts("------------------------------")

baseline_report =
  ExFairness.fairness_report(
    initial_predictions,
    initial_labels,
    initial_sensitive
  )

IO.puts("Overall Assessment: #{baseline_report.overall_assessment}")
IO.puts("Metrics Passed: #{baseline_report.passed_count}/#{baseline_report.total_count}")

IO.puts("\nDetailed Results:")

IO.puts(
  "  Demographic Parity: #{if baseline_report.demographic_parity.passes, do: "✓ PASS", else: "✗ FAIL"} (disparity: #{Float.round(baseline_report.demographic_parity.disparity * 100, 1)}%)"
)

IO.puts(
  "  Equalized Odds: #{if baseline_report.equalized_odds.passes, do: "✓ PASS", else: "✗ FAIL"}"
)

IO.puts(
  "  Equal Opportunity: #{if baseline_report.equal_opportunity.passes, do: "✓ PASS", else: "✗ FAIL"}"
)

IO.puts(
  "  Predictive Parity: #{if baseline_report.predictive_parity.passes, do: "✓ PASS", else: "✗ FAIL"}"
)

# 2.2: Legal compliance check
IO.puts("\n2.2: Legal Compliance Check (EEOC 80% Rule)")
IO.puts("---------------------------------------------")

disparate_impact =
  ExFairness.Detection.DisparateImpact.detect(
    initial_predictions,
    initial_sensitive
  )

IO.puts("Group A approval rate: #{Float.round(disparate_impact.group_a_rate * 100, 1)}%")
IO.puts("Group B approval rate: #{Float.round(disparate_impact.group_b_rate * 100, 1)}%")
IO.puts("Ratio: #{Float.round(disparate_impact.ratio, 2)}")
IO.puts("80% Rule: #{if disparate_impact.passes_80_percent_rule, do: "✓ PASS", else: "⚠ FAIL"}")

# 2.3: Document findings
IO.puts("\n2.3: Summary of Findings")
IO.puts("------------------------")

bias_detected = baseline_report.failed_count > 0 or not disparate_impact.passes_80_percent_rule

if bias_detected do
  IO.puts("⚠ BIAS DETECTED - Action required")
  IO.puts("\nIssues identified:")

  if not baseline_report.demographic_parity.passes do
    IO.puts("  • Demographic parity violation: Unequal approval rates")
  end

  if not baseline_report.equalized_odds.passes do
    IO.puts("  • Equalized odds violation: Unequal error rates")
  end

  if not baseline_report.equal_opportunity.passes do
    IO.puts("  • Equal opportunity violation: Qualified applicants treated differently")
  end

  if not baseline_report.predictive_parity.passes do
    IO.puts("  • Predictive parity violation: Approvals don't mean same thing")
  end

  if not disparate_impact.passes_80_percent_rule do
    IO.puts("  • Legal risk: Fails EEOC 80% rule")
  end

  IO.puts("\nProceeding to mitigation...")
else
  IO.puts("✓ No significant bias detected")
  IO.puts("System appears fair - no mitigation needed")
end

# ============================================================================
# STEP 3: BIAS MITIGATION
# ============================================================================

IO.puts("\n")
IO.puts("=" |> String.duplicate(60))
IO.puts("STEP 3: BIAS MITIGATION")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

# 3.1: Compute fairness weights
IO.puts("3.1: Computing Fairness Weights")
IO.puts("--------------------------------")

fairness_weights =
  ExFairness.Mitigation.Reweighting.compute_weights(
    initial_labels,
    initial_sensitive,
    target: :demographic_parity
  )

IO.puts("Weights computed successfully")
weights_list = Nx.to_flat_list(fairness_weights)
IO.puts("  Min weight: #{Float.round(Enum.min(weights_list), 3)}")
IO.puts("  Max weight: #{Float.round(Enum.max(weights_list), 3)}")
IO.puts("  Mean weight: #{Float.round(Enum.sum(weights_list) / length(weights_list), 3)}")

# 3.2: Simulate model retraining
IO.puts("\n3.2: Retraining Model with Fairness Weights")
IO.puts("--------------------------------------------")

IO.puts("(In production: Use weights in your ML training algorithm)")
IO.puts("Simulating improved model with more balanced predictions...")

# Simulate retraining with more balanced predictions
{improved_a_preds, improved_a_labels} = generate_loan_data.(200, 0.50, 0.75)
{improved_b_preds, improved_b_labels} = generate_loan_data.(200, 0.50, 0.75)

improved_predictions = Nx.tensor(improved_a_preds ++ improved_b_preds)
improved_labels = Nx.tensor(improved_a_labels ++ improved_b_labels)

improved_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {200}),
    Nx.broadcast(1, {200})
  ])

IO.puts("✓ Model retrained with fairness constraints")

# ============================================================================
# STEP 4: VALIDATION
# ============================================================================

IO.puts("\n")
IO.puts("=" |> String.duplicate(60))
IO.puts("STEP 4: VALIDATION AND COMPARISON")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

# 4.1: Test improved model
IO.puts("4.1: Testing Improved Model")
IO.puts("----------------------------")

improved_report =
  ExFairness.fairness_report(
    improved_predictions,
    improved_labels,
    improved_sensitive
  )

improved_di =
  ExFairness.Detection.DisparateImpact.detect(
    improved_predictions,
    improved_sensitive
  )

IO.puts("Improved Model Results:")
IO.puts("  Metrics Passed: #{improved_report.passed_count}/#{improved_report.total_count}")
IO.puts("  80% Rule: #{if improved_di.passes_80_percent_rule, do: "✓ PASS", else: "✗ FAIL"}")

# 4.2: Compare before and after
IO.puts("\n4.2: Before/After Comparison")
IO.puts("----------------------------")

IO.puts("\nFairness Metrics:")
IO.puts("                          Before    After")

IO.puts(
  "  Demographic Parity:     #{if baseline_report.demographic_parity.passes, do: "✓", else: "✗"}         #{if improved_report.demographic_parity.passes, do: "✓", else: "✗"}"
)

IO.puts(
  "  Equalized Odds:         #{if baseline_report.equalized_odds.passes, do: "✓", else: "✗"}         #{if improved_report.equalized_odds.passes, do: "✓", else: "✗"}"
)

IO.puts(
  "  Equal Opportunity:      #{if baseline_report.equal_opportunity.passes, do: "✓", else: "✗"}         #{if improved_report.equal_opportunity.passes, do: "✓", else: "✗"}"
)

IO.puts(
  "  Predictive Parity:      #{if baseline_report.predictive_parity.passes, do: "✓", else: "✗"}         #{if improved_report.predictive_parity.passes, do: "✓", else: "✗"}"
)

IO.puts("\nDisparity Reduction:")

IO.puts(
  "  Demographic Parity: #{Float.round(baseline_report.demographic_parity.disparity * 100, 1)}% → #{Float.round(improved_report.demographic_parity.disparity * 100, 1)}%"
)

IO.puts(
  "  80% Rule Ratio: #{Float.round(disparate_impact.ratio, 2)} → #{Float.round(improved_di.ratio, 2)}"
)

# 4.3: Recommendation
IO.puts("\n4.3: Recommendation")
IO.puts("-------------------")

if improved_report.passed_count >= baseline_report.passed_count and
     improved_di.passes_80_percent_rule do
  IO.puts("✓ RECOMMENDATION: Deploy improved model")
  IO.puts("  • Significant fairness improvement achieved")
  IO.puts("  • Legal compliance satisfied")
  IO.puts("  • Ready for production deployment")
else
  IO.puts("⚠ RECOMMENDATION: Further iteration needed")
  IO.puts("  • Some fairness issues remain")
  IO.puts("  • Consider alternative mitigation strategies")
end

# ============================================================================
# STEP 5: DOCUMENTATION
# ============================================================================

IO.puts("\n")
IO.puts("=" |> String.duplicate(60))
IO.puts("STEP 5: DOCUMENTATION")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

# Export reports
markdown_report = ExFairness.Report.to_markdown(improved_report)
File.write!("/tmp/loan_approval_fairness_report.md", markdown_report)
IO.puts("✓ Markdown report: /tmp/loan_approval_fairness_report.md")

json_report = ExFairness.Report.to_json(improved_report)
File.write!("/tmp/loan_approval_fairness_report.json", json_report)
IO.puts("✓ JSON report: /tmp/loan_approval_fairness_report.json")

# ============================================================================
# STEP 6: DEPLOYMENT CHECKLIST
# ============================================================================

IO.puts("\n")
IO.puts("=" |> String.duplicate(60))
IO.puts("STEP 6: PRE-DEPLOYMENT CHECKLIST")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

checklist = [
  {"Fairness metrics evaluated", true},
  {"Legal compliance verified", improved_di.passes_80_percent_rule},
  {"Mitigation applied", true},
  {"Validation complete", true},
  {"Documentation generated", true},
  {"Monitoring setup designed", true},
  {"Stakeholder approval", false},
  {"A/B testing plan ready", false}
]

Enum.each(checklist, fn {item, done} ->
  IO.puts("  #{if done, do: "✓", else: "☐"} #{item}")
end)

ready_for_deployment = Enum.count(checklist, fn {_, done} -> done end) >= 6

IO.puts("\n")

if ready_for_deployment do
  IO.puts("✓ SYSTEM APPROACHING DEPLOYMENT READINESS")
  IO.puts("  (Complete remaining checklist items)")
else
  IO.puts("⚠ Complete remaining checklist items before deployment")
end

# ============================================================================
# SUMMARY
# ============================================================================

IO.puts("\n")
IO.puts("=" |> String.duplicate(60))
IO.puts("WORKFLOW SUMMARY")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

IO.puts("""
This workflow demonstrated:

1. Initial Data Analysis
   ✓ Dataset preparation and understanding

2. Comprehensive Bias Detection
   ✓ Multiple fairness metrics
   ✓ Legal compliance checking
   ✓ Systematic evaluation

3. Bias Mitigation
   ✓ Reweighting implementation
   ✓ Model improvement

4. Rigorous Validation
   ✓ Before/after comparison
   ✓ Improvement verification

5. Documentation
   ✓ Report generation (Markdown & JSON)

6. Deployment Readiness
   ✓ Pre-deployment checklist

Key Takeaways:
• Fairness is a continuous process, not a one-time check
• Multiple metrics provide comprehensive assessment
• Mitigation techniques can significantly improve fairness
• Documentation and monitoring are essential
• Legal compliance must be verified

Remember: Always:
• Understand your specific use case
• Choose appropriate metrics
• Consider legal requirements
• Monitor continuously in production
""")

IO.puts("\n=== Workflow Complete ===\n")
