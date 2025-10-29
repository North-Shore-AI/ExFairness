#!/usr/bin/env elixir

# Disparate Impact Detection Example
# This example demonstrates how to use the EEOC 80% rule to detect
# adverse impact in selection decisions.

IO.puts("\n=== Disparate Impact Detection (EEOC 80% Rule) ===\n")

# The 80% rule (also called the 4/5ths rule) is a legal guideline from the
# U.S. Equal Employment Opportunity Commission. It states that if the
# selection rate for any group is less than 80% of the highest selection rate,
# this constitutes evidence of adverse impact.
#
# Formula: Ratio = min(rate_A, rate_B) / max(rate_A, rate_B)
# Legal Standard: Ratio >= 0.8 (PASS), Ratio < 0.8 (FAIL)

IO.puts("Legal Context: EEOC Uniform Guidelines on Employee Selection (1978)")
IO.puts("")

# Example 1: System that passes 80% rule
IO.puts("Example 1: Compliant Hiring System")
IO.puts("-----------------------------------")

# Group A: 50% selection rate
# Group B: 45% selection rate
# Ratio: 45/50 = 0.90 > 0.8 (PASS)

predictions = Nx.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

result = ExFairness.Detection.DisparateImpact.detect(predictions, sensitive_attr)

IO.puts("Group A selection rate: #{Float.round(result.group_a_rate * 100, 1)}%")
IO.puts("Group B selection rate: #{Float.round(result.group_b_rate * 100, 1)}%")
IO.puts("Ratio: #{Float.round(result.ratio, 2)}")
IO.puts("80% Rule: #{if result.passes_80_percent_rule, do: "✓ PASS", else: "✗ FAIL"}")
IO.puts("\nInterpretation:")
IO.puts(result.interpretation)

# Example 2: System that fails 80% rule
IO.puts("\n\nExample 2: Non-Compliant Hiring System")
IO.puts("---------------------------------------")

# Group A: 80% selection rate
# Group B: 20% selection rate
# Ratio: 20/80 = 0.25 < 0.8 (FAIL)

biased_predictions = Nx.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
sensitive_attr = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

biased_result = ExFairness.Detection.DisparateImpact.detect(biased_predictions, sensitive_attr)

IO.puts("Group A selection rate: #{Float.round(biased_result.group_a_rate * 100, 1)}%")
IO.puts("Group B selection rate: #{Float.round(biased_result.group_b_rate * 100, 1)}%")
IO.puts("Ratio: #{Float.round(biased_result.ratio, 2)}")
IO.puts("80% Rule: #{if biased_result.passes_80_percent_rule, do: "✓ PASS", else: "⚠ FAIL"}")
IO.puts("\nInterpretation:")
IO.puts(biased_result.interpretation)

IO.puts("""

⚠ LEGAL WARNING:
This system may violate EEOC guidelines and expose the organization to
discrimination claims. Consult with legal counsel immediately.
""")

# Example 3: Understanding the 80% rule
IO.puts("\n\nExample 3: Understanding the 80% Rule")
IO.puts("--------------------------------------")

IO.puts("""
The EEOC 80% Rule:

What it is:
  - Legal guideline for detecting adverse impact
  - Used in employment, lending, housing decisions
  - Established in Uniform Guidelines on Employee Selection (1978)

How it works:
  1. Calculate selection rate for each group
  2. Find the ratio: (lower rate) / (higher rate)
  3. If ratio < 0.8, there's evidence of adverse impact

Legal implications:
  - Not absolute proof of discrimination
  - Creates prima facie case (initial evidence)
  - Burden shifts to employer to show business necessity
  - Statistical significance should also be considered

Example calculations:
  Group A: 60 hired out of 100 = 60% selection rate
  Group B: 40 hired out of 100 = 40% selection rate
  Ratio: 40/60 = 0.67 < 0.8 → FAILS (adverse impact detected)

  Group A: 60 hired out of 100 = 60% selection rate
  Group B: 50 hired out of 100 = 50% selection rate
  Ratio: 50/60 = 0.83 > 0.8 → PASSES (no adverse impact)
""")

# Example 4: Real-world hiring audit
IO.puts("\n\nExample 4: Real-World Hiring Audit (1000 applicants)")
IO.puts("------------------------------------------------------")

:rand.seed(:exsplus, {100, 200, 300})

# Simulate hiring decisions with slight bias
# Group A: 55% hiring rate
# Group B: 50% hiring rate (should pass 80% rule: 50/55 = 0.91)

generate_hiring = fn group_size, hire_rate ->
  for _ <- 1..group_size, do: if(:rand.uniform() < hire_rate, do: 1, else: 0)
end

group_a_hires = generate_hiring.(500, 0.55)
group_b_hires = generate_hiring.(500, 0.50)

hire_predictions = Nx.tensor(group_a_hires ++ group_b_hires)

hire_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {500}),
    Nx.broadcast(1, {500})
  ])

hire_result = ExFairness.Detection.DisparateImpact.detect(hire_predictions, hire_sensitive)

IO.puts("Total applicants: 1000 (500 per group)")
IO.puts("")
IO.puts("Group A: #{Float.round(hire_result.group_a_rate * 100, 1)}% hired")
IO.puts("Group B: #{Float.round(hire_result.group_b_rate * 100, 1)}% hired")
IO.puts("Ratio: #{Float.round(hire_result.ratio, 2)}")
IO.puts("80% Rule: #{if hire_result.passes_80_percent_rule, do: "✓ PASS", else: "✗ FAIL"}")

if hire_result.passes_80_percent_rule do
  IO.puts("\n✓ No evidence of adverse impact")
  IO.puts("System appears compliant with EEOC guidelines")
else
  IO.puts("\n⚠ Evidence of adverse impact detected")
  IO.puts("Further investigation and mitigation required")
end

# Example 5: Problematic scenario
IO.puts("\n\nExample 5: Severe Disparate Impact")
IO.puts("-----------------------------------")

group_a_severe = generate_hiring.(500, 0.75)
group_b_severe = generate_hiring.(500, 0.35)

severe_predictions = Nx.tensor(group_a_severe ++ group_b_severe)

severe_sensitive =
  Nx.concatenate([
    Nx.broadcast(0, {500}),
    Nx.broadcast(1, {500})
  ])

severe_result = ExFairness.Detection.DisparateImpact.detect(severe_predictions, severe_sensitive)

IO.puts("Group A: #{Float.round(severe_result.group_a_rate * 100, 1)}% hired")
IO.puts("Group B: #{Float.round(severe_result.group_b_rate * 100, 1)}% hired")
IO.puts("Ratio: #{Float.round(severe_result.ratio, 2)}")
IO.puts("80% Rule: #{if severe_result.passes_80_percent_rule, do: "✓ PASS", else: "✗ FAIL"}")

IO.puts("""

🚨 CRITICAL: Severe disparate impact!
   Ratio of #{Float.round(severe_result.ratio, 2)} is far below 0.8 threshold
   This represents strong evidence of discrimination
   IMMEDIATE ACTION REQUIRED:
     1. Halt current hiring process
     2. Consult legal counsel
     3. Audit selection criteria
     4. Implement fairness interventions
     5. Document all remediation steps
""")

# Example 6: Use cases and applications
IO.puts("\n\nExample 6: When to Use the 80% Rule")
IO.puts("------------------------------------")

IO.puts("""
The 80% rule applies to:

Employment Decisions:
  - Hiring and recruitment
  - Promotions and transfers
  - Performance evaluations
  - Layoffs and terminations
  - Training opportunities

Lending Decisions:
  - Loan approvals (ECOA compliance)
  - Credit limits
  - Interest rates
  - Mortgage applications

Housing:
  - Rental applications
  - Housing sales
  - Property insurance

Education:
  - College admissions
  - Scholarship awards
  - Program selection

Other:
  - Government benefits
  - Insurance underwriting
  - Any decision affecting protected classes
""")

# Example 7: Limitations and considerations
IO.puts("\n\nExample 7: Limitations of the 80% Rule")
IO.puts("---------------------------------------")

IO.puts("""
Important Limitations:

1. Not Absolute Proof:
   - 80% rule is a guideline, not a legal requirement
   - Courts consider other factors
   - Statistical significance matters

2. Sample Size:
   - Small samples can have unreliable ratios
   - Recommended minimum: 30-50 per group
   - Use confidence intervals for small samples

3. Business Necessity Defense:
   - Employers can defend practices with business necessity
   - Must show selection criteria are job-related
   - Less discriminatory alternatives should be considered

4. Multiple Protected Classes:
   - Must check all protected groups
   - Intersectional analysis may be needed
   - Most disadvantaged group determines compliance

5. Practical Significance:
   - Very small disparities may be statistically significant
   - But may not be practically significant
   - Context and magnitude matter

6. Temporal Variation:
   - Single snapshot may be misleading
   - Monitor trends over time
   - Seasonal effects should be considered
""")

# Example 8: Integration with comprehensive fairness audit
IO.puts("\n\nExample 8: Comprehensive Compliance Check")
IO.puts("------------------------------------------")

IO.puts("\nRunning both 80% rule check and fairness metrics:")

# Generate test data
test_predictions = Nx.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
test_labels = Nx.tensor([1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1])
test_sensitive = Nx.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Legal compliance check
di_check = ExFairness.Detection.DisparateImpact.detect(test_predictions, test_sensitive)

IO.puts(
  "Legal Compliance (80% Rule): #{if di_check.passes_80_percent_rule, do: "✓ PASS", else: "✗ FAIL"}"
)

IO.puts("  Ratio: #{Float.round(di_check.ratio, 2)}")

# Fairness metrics check
fairness_check = ExFairness.fairness_report(test_predictions, test_labels, test_sensitive)
IO.puts("\nFairness Metrics: #{fairness_check.passed_count}/#{fairness_check.total_count} passed")

IO.puts("""

Recommendation:
  - Always check BOTH legal compliance (80% rule) AND fairness metrics
  - Legal compliance is necessary but not sufficient
  - Fairness metrics provide deeper insight into bias
  - Document all checks for audit trail
""")

IO.puts("\n=== Example Complete ===\n")
