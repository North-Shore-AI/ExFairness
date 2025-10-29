#!/usr/bin/env elixir

# Comprehensive Fairness Report Example
# This example demonstrates how to generate and export comprehensive
# fairness reports that evaluate multiple metrics at once.

IO.puts("\n=== Comprehensive Fairness Report Example ===\n")

IO.puts("Scenario: Loan approval system audit")
IO.puts("")

# Generate realistic loan approval data
:rand.seed(:exsplus, {42, 42, 42})

# Simulate 200 loan applications
# Group A: 60% approval rate, 70% of approved are good loans
# Group B: 50% approval rate, 70% of approved are good loans
# This creates demographic disparity but maintains predictive parity

generate_loan_data = fn group_size, approval_rate, success_rate ->
  # Generate predictions (approval decisions)
  predictions = for _ <- 1..group_size, do: if(:rand.uniform() < approval_rate, do: 1, else: 0)

  # Generate labels (whether loan would be successful)
  labels =
    Enum.map(predictions, fn pred ->
      case pred do
        1 -> if :rand.uniform() < success_rate, do: 1, else: 0
        # Some rejected would have succeeded
        0 -> if :rand.uniform() < 0.3, do: 1, else: 0
      end
    end)

  {predictions, labels}
end

{group_a_preds, group_a_labels} = generate_loan_data.(100, 0.60, 0.70)
{group_b_preds, group_b_labels} = generate_loan_data.(100, 0.50, 0.70)

predictions = Nx.tensor(group_a_preds ++ group_b_preds)
labels = Nx.tensor(group_a_labels ++ group_b_labels)

sensitive_attr =
  Nx.concatenate([
    Nx.broadcast(0, {100}),
    Nx.broadcast(1, {100})
  ])

# Example 1: Generate comprehensive report
IO.puts("Example 1: Generate Comprehensive Report")
IO.puts("------------------------------------------")

report = ExFairness.fairness_report(predictions, labels, sensitive_attr)

IO.puts("\n" <> report.overall_assessment)
IO.puts("\nMetrics Summary:")
IO.puts("  Total metrics evaluated: #{report.total_count}")
IO.puts("  Passed: #{report.passed_count}")
IO.puts("  Failed: #{report.failed_count}")

IO.puts("\n--- Demographic Parity ---")
dp = report.demographic_parity
IO.puts("  Group A rate: #{Float.round(dp.group_a_rate * 100, 1)}%")
IO.puts("  Group B rate: #{Float.round(dp.group_b_rate * 100, 1)}%")
IO.puts("  Disparity: #{Float.round(dp.disparity * 100, 1)}%")
IO.puts("  Status: #{if dp.passes, do: "✓ PASS", else: "✗ FAIL"}")

IO.puts("\n--- Equalized Odds ---")
eo = report.equalized_odds
IO.puts("  Group A TPR: #{Float.round(eo.group_a_tpr * 100, 1)}%")
IO.puts("  Group B TPR: #{Float.round(eo.group_b_tpr * 100, 1)}%")
IO.puts("  TPR Disparity: #{Float.round(eo.tpr_disparity * 100, 1)}%")
IO.puts("  Group A FPR: #{Float.round(eo.group_a_fpr * 100, 1)}%")
IO.puts("  Group B FPR: #{Float.round(eo.group_b_fpr * 100, 1)}%")
IO.puts("  FPR Disparity: #{Float.round(eo.fpr_disparity * 100, 1)}%")
IO.puts("  Status: #{if eo.passes, do: "✓ PASS", else: "✗ FAIL"}")

IO.puts("\n--- Equal Opportunity ---")
eop = report.equal_opportunity
IO.puts("  Group A TPR: #{Float.round(eop.group_a_tpr * 100, 1)}%")
IO.puts("  Group B TPR: #{Float.round(eop.group_b_tpr * 100, 1)}%")
IO.puts("  Disparity: #{Float.round(eop.disparity * 100, 1)}%")
IO.puts("  Status: #{if eop.passes, do: "✓ PASS", else: "✗ FAIL"}")

IO.puts("\n--- Predictive Parity ---")
pp = report.predictive_parity
IO.puts("  Group A PPV: #{Float.round(pp.group_a_ppv * 100, 1)}%")
IO.puts("  Group B PPV: #{Float.round(pp.group_b_ppv * 100, 1)}%")
IO.puts("  Disparity: #{Float.round(pp.disparity * 100, 1)}%")
IO.puts("  Status: #{if pp.passes, do: "✓ PASS", else: "✗ FAIL"}")

# Example 2: Export report to Markdown
IO.puts("\n\nExample 2: Export Report to Markdown")
IO.puts("-------------------------------------")

markdown_report = ExFairness.Report.to_markdown(report)
File.write!("/tmp/fairness_report.md", markdown_report)
IO.puts("Markdown report saved to: /tmp/fairness_report.md")
IO.puts("\nFirst 500 characters of markdown report:")
IO.puts(String.slice(markdown_report, 0..499) <> "...")

# Example 3: Export report to JSON
IO.puts("\n\nExample 3: Export Report to JSON")
IO.puts("---------------------------------")

json_report = ExFairness.Report.to_json(report)
File.write!("/tmp/fairness_report.json", json_report)
IO.puts("JSON report saved to: /tmp/fairness_report.json")
IO.puts("\nFirst 500 characters of JSON report:")
IO.puts(String.slice(json_report, 0..499) <> "...")

# Example 4: Custom metric selection
IO.puts("\n\nExample 4: Custom Metric Selection")
IO.puts("-----------------------------------")

IO.puts("\nEvaluating only demographic parity and equal opportunity:")

custom_report =
  ExFairness.fairness_report(
    predictions,
    labels,
    sensitive_attr,
    metrics: [:demographic_parity, :equal_opportunity]
  )

IO.puts("Total metrics evaluated: #{custom_report.total_count}")
IO.puts("Passed: #{custom_report.passed_count}")
IO.puts("Failed: #{custom_report.failed_count}")
IO.puts("\n" <> custom_report.overall_assessment)

# Example 5: Fair system comparison
IO.puts("\n\nExample 5: Comparing Fair vs Biased Systems")
IO.puts("--------------------------------------------")

# Generate FAIR system data
{fair_a_preds, fair_a_labels} = generate_loan_data.(100, 0.55, 0.70)
{fair_b_preds, fair_b_labels} = generate_loan_data.(100, 0.55, 0.70)

fair_predictions = Nx.tensor(fair_a_preds ++ fair_b_preds)
fair_labels = Nx.tensor(fair_a_labels ++ fair_b_labels)

fair_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {100}),
    Nx.broadcast(1, {100})
  ])

fair_report = ExFairness.fairness_report(fair_predictions, fair_labels, fair_sensitive)

IO.puts("\nOriginal System:")
IO.puts("  " <> report.overall_assessment)

IO.puts("\nImproved Fair System:")
IO.puts("  " <> fair_report.overall_assessment)

IO.puts("\nComparison:")
IO.puts("  Original: #{report.passed_count}/#{report.total_count} metrics passed")
IO.puts("  Improved: #{fair_report.passed_count}/#{fair_report.total_count} metrics passed")

IO.puts(
  "  Improvement: #{fair_report.passed_count - report.passed_count} additional metrics passed"
)

# Example 6: Custom threshold
IO.puts("\n\nExample 6: Stricter Fairness Thresholds")
IO.puts("----------------------------------------")

strict_report =
  ExFairness.fairness_report(
    predictions,
    labels,
    sensitive_attr,
    # 5% instead of default 10%
    threshold: 0.05
  )

IO.puts("With 10% threshold: #{report.passed_count}/#{report.total_count} passed")
IO.puts("With 5% threshold: #{strict_report.passed_count}/#{strict_report.total_count} passed")
IO.puts("\nStricter thresholds help ensure higher fairness standards")

# Example 7: Production monitoring scenario
IO.puts("\n\nExample 7: Production Monitoring Workflow")
IO.puts("------------------------------------------")

IO.puts("""
Recommended workflow for production systems:

1. BASELINE AUDIT
   - Generate comprehensive fairness report on training data
   - Export to JSON for version control
   - Document any known fairness issues

2. PRE-DEPLOYMENT CHECK
   - Run fairness report on test set
   - Ensure all critical metrics pass
   - Review interpretations carefully

3. POST-DEPLOYMENT MONITORING
   - Schedule periodic fairness audits (daily/weekly)
   - Track metrics over time
   - Alert on fairness degradation

4. INCIDENT RESPONSE
   - If metrics fail, investigate immediately
   - Generate detailed report for stakeholders
   - Apply mitigation techniques
   - Re-audit after fixes

5. COMPLIANCE DOCUMENTATION
   - Export markdown reports for human review
   - Store JSON reports for audit trail
   - Include in model cards and documentation
""")

# Generate sample monitoring report
monitoring_report = ExFairness.fairness_report(predictions, labels, sensitive_attr)
timestamp = DateTime.utc_now() |> DateTime.to_string()

IO.puts("\n--- Sample Monitoring Report ---")
IO.puts("Timestamp: #{timestamp}")
IO.puts("System: Loan Approval Model v1.2")
IO.puts("Sample Size: 200")
IO.puts("\n" <> monitoring_report.overall_assessment)

if monitoring_report.failed_count > 0 do
  IO.puts("\n⚠ ALERT: Fairness issues detected!")
  IO.puts("Action required: Review failed metrics and apply mitigation")
else
  IO.puts("\n✓ All fairness checks passed")
end

# Example 8: Understanding the report structure
IO.puts("\n\nExample 8: Understanding Report Structure")
IO.puts("------------------------------------------")

IO.puts("""
Report Structure:
  - demographic_parity: Map with rates, disparity, passes, interpretation
  - equalized_odds: Map with TPR, FPR, disparities, passes, interpretation
  - equal_opportunity: Map with TPR, disparity, passes, interpretation
  - predictive_parity: Map with PPV, disparity, passes, interpretation
  - overall_assessment: Summary string
  - passed_count: Number of metrics that passed
  - failed_count: Number of metrics that failed
  - total_count: Total metrics evaluated

Each metric map includes:
  - Numeric values (rates, disparities)
  - Boolean pass/fail status
  - Human-readable interpretation
""")

IO.puts("\n=== Example Complete ===\n")
